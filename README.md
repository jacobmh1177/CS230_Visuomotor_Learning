# CS230_Visuomotor_Learning

Add the dataset into the parser/ on your local directory. In parse_data.py update path.data_folder if necessary then run parse_data.py. 

parse_data.py will have to be modified to save the data in mini-batches because the entire dataset is too large to hold in memory.

## Setup

Download the `TeleOpVRSession_2018-02-05_15-44-11.zip` file from our Google Drive into the `datasets` directory. Unzip it, so that there should now be a `datasets/TeleOpVRSession_2018-02-05_15-44-11/` directory with a `_SessionStateData.proto` and a bunch of jpg images inside. Update the `path.data_folder` variable in `parse_data.py` to be `datasets/TeleOpVRSession_2018-02-05_15-44-11/`.

