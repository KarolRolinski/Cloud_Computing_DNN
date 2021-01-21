#!/usr/bin/env python
# coding: utf-8

# In[46]:


import boto3, re
from sagemaker import get_execution_role

role = get_execution_role()

import tensorflow.keras
from tensorflow.keras.models import model_from_json

#!mkdir keras_model


# In[47]:


get_ipython().system('ls keras_model')


# In[55]:


import tensorflow as tf

json_file = open('./keras_model/'+'uci3_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={"Adam": tf.keras.optimizers.Adam(learning_rate=0.0008),
                                                                 "L2": tf.keras.regularizers.l2(0.001)})
loaded_model.load_weights('./keras_model/uci3_model_weights.h5')


print("Gotowe")


# In[56]:


from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

if tf.executing_eagerly():
   tf.compat.v1.disable_eager_execution()

model_version = '1'
export_dir = 'export/Servo/' + model_version

build = builder.SavedModelBuilder(export_dir)
signature = predict_signature_def(inputs={"inputs": loaded_model.input}, outputs={"score": loaded_model.output})


# In[59]:


from keras import backend as K

import tensorflow.compat.v1.keras.backend as K

with K.get_session() as sess:
    build.add_meta_graph_and_variables(
        sess=sess, tags=[tag_constants.SERVING], signature_def_map={"serving_default": signature})
    build.save()


# In[63]:


get_ipython().system('ls export/Servo/1/variables')


# In[64]:


import tarfile
with tarfile.open('model.tar.gz', mode='w:gz') as archive:
    archive.add('export', recursive=True)


# In[65]:


import sagemaker
sagemaker_session = sagemaker.Session()
inputs = sagemaker_session.upload_data(path='model.tar.gz', key_prefix='model')


# In[66]:


get_ipython().system('touch train.py')


# In[74]:


from sagemaker.tensorflow.model import TensorFlowModel
sagemaker_model = TensorFlowModel(model_data = 's3://' + sagemaker_session.default_bucket() + '/model/model.tar.gz',
                                  role = role,
                                  framework_version = '2.2.0',
                                  entry_point = 'train.py')


# In[75]:


get_ipython().run_cell_magic('time', '', "predictor = sagemaker_model.deploy(initial_instance_count=1,\n                                  instance_type='ml.t2.medium')")


# In[77]:


predictor.endpoint


# In[78]:


endpoint_name = 'tensorflow-inference-2021-01-14-13-59-27-071'


# In[80]:


import sagemaker
from sagemaker.tensorflow.model import TensorFlowModel
predictor=sagemaker.tensorflow.model.TensorFlowPredictor(endpoint_name, sagemaker_session)


# In[99]:


import json
import boto3
import numpy as np
import io

client = boto3.client('runtime.sagemaker')

ds_test = [[1309.492,1372.592,6649.217,8.741,471.177,547.352,1.14,6.11,1.019,2.172,0.096]]


result = predictor.predict(ds_test)
print(result)


# In[100]:


sagemaker.Session().delete_endpoint(predictor.endpoint_name)
print(f"deleted {predictor.endpoint_name} successfully!")


