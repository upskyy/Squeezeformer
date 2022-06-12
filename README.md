# Squeezeformer


<p  align="center"> 
     <a href="https://github.com/sooftware/jasper/blob/main/LICENSE">
          <img src="http://img.shields.io/badge/license-Apache--2.0-informational"> 
     </a>
     <a href="https://github.com/pytorch/pytorch">
          <img src="http://img.shields.io/badge/framework-PyTorch-informational"> 
     </a>
     <a href="https://www.python.org/dev/peps/pep-0008/">
          <img src="http://img.shields.io/badge/codestyle-PEP--8-informational"> 
     </a>
     <a href="https://github.com/sooftware/conformer">
          <img src="http://img.shields.io/badge/build-passing-success"> 
     </a>

  
Squeezeformer incorporates the Temporal U-Net structure, which reduces the cost of the
multi-head attention modules on long sequences, and a simpler block structure of feed-forward module,
followed up by multi-head attention or convolution modules,
instead of the Macaron structure proposed in Conformer.  

<img width="417" alt="스크린샷 2022-06-11 오전 1 19 40" src="https://user-images.githubusercontent.com/54731898/173109027-76a51857-b3cf-4616-938d-d3b990a4cf13.png">  

This repository contains only model code, but you can train with squeezeformer at [openspeech](https://github.com/openspeech-team/openspeech/blob/main/openspeech/models/squeezeformer/model.py).  
     
     
  

## Installation
```   
pip install squeezeformer  
```   
            
     
## Usage
```python
import torch
import torch.nn as nn
from squeezeformer.model import Squeezeformer


BATCH_SIZE = 4
SEQ_LENGTH = 500
INPUT_SIZE = 80
NUM_CLASSES = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CTCLoss().to(device)
model = Squeezeformer(
     num_classes=NUM_CLASSES,
).to(device)

inputs = torch.FloatTensor(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE).to(device)
input_lengths = torch.IntTensor([500, 450, 400, 350]).to(device)
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                           [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                           [1, 3, 3, 3, 3, 3, 4, 2, 0, 0],
                           [1, 3, 3, 3, 3, 3, 6, 2, 0, 0]]).to(device)
target_lengths = torch.LongTensor([9, 8, 7, 7]).to(device)

# Forward propagate
outputs, output_lengths = model(inputs, input_lengths)

# Calculate CTC Loss
for _ in range(3):
     outputs, output_lengths = model(inputs, input_lengths)
     loss = criterion(outputs.transpose(0, 1), targets[:, 1:], output_lengths, target_lengths)
     loss.backward()
```
  
     
## Reference
- [kssteven418/Squeezeformer](https://github.com/kssteven418/Squeezeformer)  
- [sooftware/conformer](https://github.com/sooftware/conformer)
- [Squeezeformer: An Efficient Transformer for Automatic Speech Recognition](https://arxiv.org/abs/2206.00888)
  
     
## License
```
Copyright 2022 Sangchun Ha.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
