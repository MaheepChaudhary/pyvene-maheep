# Notes and Issues:

## Notes:

* We also have to insert the commands in the `__init__.py` file under `models`.

* It is required to insert the module mappings in the `intervenable_modelcard.py` (I deployed a little trick to import my model).


## ⁉️ Issues:

* Facing the issue with the import of the  `RepresentationConfig`, hence, made it directly into the file. 

```
from collections import OrderedDict, namedtuple

RepresentationConfig = namedtuple(
    "RepresentationConfig",
    "layer component unit "
    "max_number_of_units "
    "low_rank_dimension intervention_type "
    "subspace_partition group_key intervention_link_key moe_key "
    "source_representation hidden_source_representation",
    defaults=(
        0, "block_output", "pos", 1, None,
        None, None, None, None, None, None, None),
)
```

* Should we use the `AutoImageProcessor` as the tokenizer, i.e. `from transformers import AutoImageProcessor`?