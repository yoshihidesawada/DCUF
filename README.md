# DCUF
Disentangling controllable and uncontrollable factors of variation (DCUF) by Interacting with the World

# Dependencies
- python version 2.7.6
- chainer version 1.8.2

# Usage
`python main.py ${LAMBDA} ${ICF_EPOCH} ${AE_EPOCH} ${DCUF_EPOCH}`  
  
${LAMBDA}: hyparparameter to balance autoencoder (AE) and disentangled objectives  
${ICF_EPOCH}: the number of epoch to pretrain independently controllable factor's model  
${AE_EPOCH}: the number of epoch to pretrain second AE (in our papaer, we set 0)  
${DCUF_EPOCH}: the number of epoch to train DCUF model  
