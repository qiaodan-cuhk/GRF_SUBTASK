


create_task.py lin 18

# 这两行正常是注释掉的，这里为了测试test加回来
data["env_args"]["rewards"] = 'scoring, reward_test'
data["t_max"] = 2000


buffer.load 在torch 2.6以后改成 weight_only=False



parallel runner line 18

# 为了测试
self.batch_size = 1


default.yaml 修改

原始为
save_model: False # Save the models to disk
save_model_interval: 50000 # Save models after this many timesteps
修改为
save_model: True # Save the models to disk
save_model_interval: 1000 # Save models after this many timesteps