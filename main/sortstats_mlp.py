import torch
import torch.nn as nn
from diffsort import DiffSortNet


# ─────────────────────────────────────────────────────────────────────────────
# SortStatsLayer
# ─────────────────────────────────────────────────────────────────────────────

class SortStatsLayer(nn.Module):
    """
    Capa estructural de SortStats-MLP.

    Recibe un vector de caracteristicas x de forma [batch, dim] y produce
    representaciones complementarias que se concatenan con el flujo principal.
    El parametro 'branches' controla que ramas se activan. 

    Nota sobre la red de ordenamiento:
        Se usa 'bitonic' en lugar de 'odd_even' porque con vectores de alta dimension agota memoria en el backward.

        La red bitonica requiere que input_dim sea potencia de 2, por lo que
        se aplica padding con un valor muy negativo hasta la siguiente
        potencia de 2, y se descarta despues del sort.

    Args:
        input_dim : dimension real del vector de entrada.
        steepness : controla la dureza de la relajacion del sort.
                    Mayor steepness = mas parecido al sort exacto,
                    pero gradientes mas inestables. Default: 5.0.
        branches  : que ramas activar. Opciones: 'both' | 'sort_only' | 'stats_only'. Default: 'both'.
    """
    def __init__(self, input_dim, steepness=5.0, branches='both'):
        super().__init__()
        assert branches in ('both', 'sort_only', 'stats_only'), \
            f"branches debe ser 'both', 'sort_only' o 'stats_only'. Recibido: '{branches}'"

        self.input_dim  = input_dim
        self.branches   = branches
        self.padded_dim = 1 << (input_dim - 1).bit_length()  # siguiente potencia de 2
        self.pad_amount = self.padded_dim - input_dim

        # Solo se instancia el sorter si se necesita la rama de ordenamiento
        if branches in ('both', 'sort_only'):
            self.sorter = DiffSortNet('bitonic', self.padded_dim, steepness=steepness)
        else:
            self.sorter = None

        # Calcula output_dim segun las ramas activas
        if branches == 'both':
            self.output_dim = input_dim * 2 + 4
        elif branches == 'sort_only':
            self.output_dim = input_dim * 2
        else:  # stats_only
            self.output_dim = input_dim + 4

    def _apply(self, fn, recurse=True):
        """
        DiffSortNet guarda sorting_network como lista anidada de tensores
        planos (no registrados como buffers de nn.Module), por lo que
        .to(device)/.cuda()/.cpu() no los mueve automaticamente.
        Se sobreescribe _apply para que esos tensores sigan al resto del modulo.
        Solo aplica si la rama de ordenamiento esta activa.
        """
        super()._apply(fn, recurse=recurse)
        if self.sorter is not None:
            self.sorter.sorting_network = [
                [fn(t) for t in level] for level in self.sorter.sorting_network
            ]
        return self

    def _sorted(self, x):
        """Aplica el sort con padding y devuelve el vector ordenado."""
        if self.pad_amount > 0:
            pad_value            = x.min().detach() - 1e4
            x_padded             = torch.nn.functional.pad(x, (0, self.pad_amount), value=0.0)
            x_padded[:, self.input_dim:] = pad_value
        else:
            x_padded = x
        sorted_padded, _ = self.sorter(x_padded)
        return sorted_padded[:, self.pad_amount:]  # descartar padding

    def _stats(self, x):
        """Calcula los 4 descriptores estadisticos del vector."""
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std(dim=1, keepdim=True)
        min_ = x.min(dim=1, keepdim=True).values
        max_ = x.max(dim=1, keepdim=True).values
        return torch.cat([mean, std, min_, max_], dim=1)

    def forward(self, x):
        if self.branches == 'both':
            return torch.cat([x, self._sorted(x), self._stats(x)], dim=1)
        elif self.branches == 'sort_only':
            return torch.cat([x, self._sorted(x)], dim=1)
        else:  # stats_only
            return torch.cat([x, self._stats(x)], dim=1)


# SortStatsMLPClassifier

class SortStatsMLPClassifier(nn.Module):
    """
    MLP para clasificacion extendido con la
    SortStats Layer en capas configurables.

    """
    def __init__(self, input_dim, hidden_dims, num_classes,
                 dropout=0.2, activation=nn.ReLU,
                 apply_sortstats_at=None, steepness=5.0, branches='both'):
        super().__init__()
        self.num_classes        = num_classes
        self.apply_sortstats_at = set(apply_sortstats_at or [])

        dims = [input_dim] + hidden_dims

        self.sortstats_layers = nn.ModuleDict()
        self.blocks           = nn.ModuleList()

        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            block_in_dim = in_d

            if i in self.apply_sortstats_at:
                sortstats = SortStatsLayer(in_d, steepness=steepness, branches=branches)
                self.sortstats_layers[str(i)] = sortstats
                block_in_dim = sortstats.output_dim

            block_layers = [
                nn.Linear(block_in_dim, out_d),
                nn.BatchNorm1d(out_d),
                activation(),
            ]
            if dropout > 0:
                block_layers.append(nn.Dropout(dropout))

            self.blocks.append(nn.Sequential(*block_layers))

        output_dim = 1 if num_classes == 2 else num_classes
        self.output_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if str(i) in self.sortstats_layers:
                x = self.sortstats_layers[str(i)](x)
            x = block(x)
        return self.output_layer(x)

    @torch.no_grad()
    def predict(self, x):
        """Devuelve clases predichas (int)."""
        self.eval()
        logits = self(x)
        if self.num_classes == 2:
            return (torch.sigmoid(logits) >= 0.5).long().squeeze(1)
        return logits.argmax(dim=1)

    @torch.no_grad()
    def predict_proba(self, x):
        """Devuelve probabilidades por clase."""
        self.eval()
        logits = self(x)
        if self.num_classes == 2:
            prob = torch.sigmoid(logits)
            return torch.cat([1 - prob, prob], dim=1)
        return torch.softmax(logits, dim=1)


# Configuraciones por dataset
# Inicialmente se planteo utilizar diferentes, sin embargo esto generaria un sesgo en la comparacion de resultados,
# por lo que se opto por utilizar la misma configuracion para todos los datasets.

DATASET_CONFIGS = {
    'santander': {
        'hidden_dims'  : [32, 16, 8],
        'dropout'      : 0.2,
        'batch_size'   : 2048,
        'task'         : 'binary',
        'imbalanced'   : True,
        'metric_focus' : 'f1',
        'description'  : 'Santander Customer Transaction Prediction (~200K filas, 200 features)',
    },
    'diabetes': {
        'hidden_dims'  : [32, 16, 8],
        'dropout'      : 0.2,
        'batch_size'   : 2048,
        'task'         : 'binary',
        'imbalanced'   : False,
        'metric_focus' : 'f1',
        'description'  : 'Diabetes Health Indicators 50/50 (~70K filas, 21 features)',
    },
    'forest_cover_small': {
        'hidden_dims'  : [32, 16, 8],
        'dropout'      : 0.2,
        'batch_size'   : 512,
        'task'         : 'multiclass',
        'imbalanced'   : False,
        'metric_focus' : 'f1_macro',
        'description'  : 'Forest Cover Type competencia (~15K filas, 54 features, 7 clases)',
    },
    'california': {
        'hidden_dims'  : [32, 16, 8],
        'dropout'      : 0.2,
        'batch_size'   : 512,
        'task'         : 'binary',
        'imbalanced'   : False,   
        'metric_focus' : 'f1',
        'description'  : 'California Housing (~20K filas, 8 features)',
    },
    'adult': {
        'hidden_dims'  : [32, 16, 8],
        'dropout'      : 0.2,
        'batch_size'   : 1024,
        'task'         : 'binary',
        'imbalanced'   : True,    # ~75/25 aprox.
        'metric_focus' : 'f1',
        'description'  : 'Adult / Census Income (OpenML 1590, ~48K filas, 14 features mixtas)',
    },
    'electricity': {
        'hidden_dims'  : [32, 16, 8],
        'dropout'      : 0.2,
        'batch_size'   : 1024,
        'task'         : 'binary',
        'imbalanced'   : False,
        'metric_focus' : 'f1',
        'description'  : 'Electricity (OpenML, ~38K filas, 8 features)',
    },
    'miniboone': {
        'hidden_dims'  : [32, 16, 8],
        'dropout'      : 0.2,
        'batch_size'   : 1024,
        'task'         : 'binary',
        'imbalanced'   : True,    # ~72/28 aprox.
        'metric_focus' : 'f1',
        'description'  : 'MiniBooNE (OpenML task 361068, ~73K filas, 50 features)',
    },
}


# Funciones auxiliares


def build_model(dataset_name, n_features, num_classes,
                apply_sortstats_at=None, steepness=5.0, branches='both'):
    """
    Construye un SortStatsMLPClassifier con la configuracion del dataset indicado:
        dataset_name       : clave del diccionario DATASET_CONFIGS.
        n_features         : numero de features de entrada.
        num_classes        : numero de clases.
        apply_sortstats_at : lista de indices de capas donde aplicar SortStats.
                             None o [] = MLP estandar.
        steepness          : dureza de la relajacion del sort.
        branches           : ramas activas en la SortStatsLayer.

    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Dataset '{dataset_name}' no encontrado en DATASET_CONFIGS. "
            f"Opciones disponibles: {list(DATASET_CONFIGS.keys())}"
        )

    cfg = DATASET_CONFIGS[dataset_name]
    return SortStatsMLPClassifier(
        input_dim          = n_features,
        hidden_dims        = cfg['hidden_dims'],
        num_classes        = num_classes,
        dropout            = cfg['dropout'],
        activation         = nn.ReLU,
        apply_sortstats_at = apply_sortstats_at,
        steepness          = steepness,
        branches           = branches,
    )


def build_criterion(num_classes, y_train=None):
    """
    Construye la funcion de perdida apropiada para el tipo de problema.

    - Binario balanceado    : BCEWithLogitsLoss estandar.
    - Binario desbalanceado : BCEWithLogitsLoss con pos_weight calculado a partir de y_train.
    - Multiclase            : CrossEntropyLoss estandar.
    """
    if num_classes > 2:
        return nn.CrossEntropyLoss()

    if y_train is not None:
        import numpy as np
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        if n_pos > 0 and n_neg / n_pos > 1.5:  # desbalance significativo (ratio > 1.5)
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
            print(f'pos_weight aplicado: {pos_weight.item():.4f} '
                  f'(neg={n_neg}, pos={n_pos})')
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return nn.BCEWithLogitsLoss()


def free_gpu_memory(model=None, optimizer=None, scheduler=None):
    """
    Libera la memoria GPU de una corrida anterior antes de instanciar un modelo nuevo.
    """
    import gc
    for obj in [model, optimizer, scheduler]:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f'GPU memoria reservada: {torch.cuda.memory_reserved()/1e9:.2f} GB  '
              f'asignada: {torch.cuda.memory_allocated()/1e9:.2f} GB')
