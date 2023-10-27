# # import os
import pandas as pd
# # from tqdm import tqdm
# # TRAIN_DIR = "Train"
# # for folder in tqdm(os.listdir(TRAIN_DIR)):
# #     for img in tqdm(sorted(os.listdir(os.path.join(TRAIN_DIR,folder)))):
# #         print(img)
test_file = pd.DataFrame(columns=['image_name', 'label'])
# print(test_file)
test_file.at[0,'image_name'] = "snow.png"
test_file.to_csv('CNN_Predictions.csv', index=False)