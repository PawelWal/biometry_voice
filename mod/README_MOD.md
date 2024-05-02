## How to run

#### FREQUENCY
```
python modify.py --data_dir ./data --tr frequency --output_dir ./data_mod
```

#### NOISE
```
python modify.py --data_dir ./data --tr noise --output_dir ./data_mod
```

#### IRREGULAR_NOISE
```
python modify.py --data_dir ./data --tr irregular-noise --output_dir ./data_mod
```

## How to extend
1. Add to `TransformationType` new type
2. Add record to MODS dictionary
3. Add class that inherits from class Transformation
4. Add your case in `modify` method