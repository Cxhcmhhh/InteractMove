
# Structure

The detailed structure is as follows:

```bash
- afford-motion/
  - body_models/
    - smplx/
      - SMPLX_NEUTRAL.npz
      - ...
  - data/
    - custom/
    - eval/
    - InteractMove/
    - POINTTRANS_C_N8192_E300/
    - Mean_Std_*
    - ...
  - outputs/
    - CDM-Perceiver-ALL/                # pre-trained ADM model on all datasets for evaluation
    - CMDM-Enc-ALL/                     # pre-trained CMDM model on all datasets for evaluation
  - configs/
  - datasets/
  - ...
```

# Process Data

The following process is to prepare the data for training model using InteractMove Dataset.


## 1. Process motion of the dataset

### InteractMove

After download the motion parts, run the following commands:

```bash
python prepare/process.py --dataset InteractMove --data_dir ${YOUR_PATH}/InteractMove
```



## 2. Convert SMPL-X to vectorized representations (joint positions)

```bash
python prepare/smplx_to_vec.py --dataset ${DATASET}
# e.g., python prepare/smplx_to_vec.py --dataset InteractMove
```
## 3. Process the scene point cloud

```bash
python prepare/process_scene.py
```

## 4. Generate Contact Data

```bash
python prepare/generate_contact_data.py --random_segment
```

## 5. Re-split the dataset

```bash
python prepare/split.py
```
