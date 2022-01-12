##Copyright 2017 Adobe Systems Inc.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.


##############################################################
#Set your paths here

in_path = '/sdd/WangRui/dataSets/COCO2014/train2014/'

txt_path = "/sdd/WangRui/dataSets/Combined_Dataset/Training_set/training_bg_names.txt"

out_path = "/sdd/WangRui/dataSets/Combined_Dataset/my_clean/train/coco_bg"


##############################################################

import os
import shutil

import pdb

with open(txt_path, "r") as f:
    names = f.readlines()

for name in names:
    full_name = os.path.join(in_path, name).strip()
    if not os.path.exists(full_name):
        print(name, "not exist!!!!!!!")
    else:
        shutil.copyfile(full_name, os.path.join(out_path, name).strip())

print("DONE!")





