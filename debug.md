#### 固定全局onehot的训练逻辑（不采用）

根据整体target task的agent num，固定one-hot长度，分配给每个agent对应的one-hot编码
创建一个onehot-dict存储到config中：
如 5v5，A 10000，B 01000，C 00100，D 00010，E 00001
该 id 编码全程不随子团队变化而改变，保持固定，避免子团队的onehot重新分配，便于doe cls训练稳定


    
首先合并child group信息，得到target task（当前任务）的role_ids和num agents
生成本layer的config（考虑用create task替换）
读取child group的doe cls，合并得到新的doe cls，存储在 merged_doe_name，用于rl训练开始时加载给mac learner。这个doe cls的label直接load from pt，用于训练过程的参数控制。
比如从 group 6 加载了 AB，group 7 加载了 CDE
那么其加载、合并、存储的doe cls为 [6 6 7 7 7]，输入 obs 的 onehot id 保持不变
rl训练结束后需要用当前训练group id 5进行doe cls训练，得到 [5 5 5 5 5]的doe cls用于下一层

但是如何处理agent id映射？因为每次merge team的时候，agent的排序和组合都是随机的

执行run.py
    修改ckpt path不为""，以在训练初期 learner.load init team policy
    加载 merged doe name 这个cls，利用load模式的from config
    进行训练
    训练结束后存储buffer到本层folder
    todo：更改role_ids的list命名
    读取buffer进行新的cls训练，利用train模式的from config，存储为 save doe name，用于下一阶段训练
    存储final policy ckpt 到文件夹路径



todo：读取child group的policy pth，合并得到新的policy并存储到本target task下作为init policy


#### doe cls 剔除onehot的训练逻辑（采用）

具体修改：
1. run.py 添加process buffer for doe函数，剔除末尾args.agent_nums长度的onehot编码
2. non_shared_controller中 build inputs函数改为从config.total_agents设置onehot长度，该参数根据最后的target task设置，onehot编码根据当前agent nums创建。
3. 是否需要确定mlp class的input dim？


根据整体target task的agent num，固定one-hot长度，根据当前子任务的agent num，动态分配 onehot id，如subtask 5只需要3个agent，全局5 agents，那么训练该层子任务时只分配 10000 01000 00100。doe cls不包括agent id onehot，只根据obs预测label，灵活可调整。

    
首先合并child group信息，得到target task（当前任务）的role_ids和num agents
生成本layer的config（考虑用create task替换）


读取child group的doe cls，合并得到新的doe cls，存储在 merged_doe_name，用于rl训练开始时加载给mac learner。这个doe cls的label直接load from pt，用于训练过程的参数控制。加载的过程中，根据subtask存储的顺序进行merge，保证agent policy与doe cls是同一个agent。（设计assert判断）


比如从 group 6 加载了 AB，group 7 加载了 CDE，merge顺序为 [7, 6]，那么doe也对应的concate mlps [7 ckpt, 6 ckpt]，保证一致，也就是【CDE AB】，doe是【777 66】，不区分每层subtask中每个agent的具体任务。

rl训练结束后保存buffer用于doe cls训练，添加func剔除buffer obs中的onehot编码，保证训练doe cls只根据obs。

执行run.py
    修改ckpt path不为""，以在训练初期 learner.load init team policy
    加载 merged doe name 这个cls，利用load模式的from config
    进行训练
    训练结束后存储buffer到本层folder
    todo：更改role_ids的list命名
    读取buffer进行新的cls训练，利用train模式的from config，存储为 save doe name，用于下一阶段训练
    存储final policy ckpt 到文件夹路径



todo：读取child group的policy pth，合并得到新的policy并存储到本target task下作为init policy


#### 1. 地图路径问题
```from gfootball import``` 直接导向了```anaconda/lib/gfootball```，所以把scenarios文件复制过去就好了，只出现在dan的server上

#### 2. 临时缩小训练步骤

源代码中这两行是注释掉的，这里为了测试test加速child训练临时加上

```create_task.py lin 21-23```

data["env_args"]["rewards"] = 'scoring, reward_test'
data["t_max"] = 2000

#### 3. torch.load 报错
```torch.load(xxx)``` 在torch 2.6以后需要改成 ```torch.load(xxx, weight_only=False)```

#### 4. 为了测试debug
关闭并行环境启动，设置
```parallel_runner.py``` 中 line 18 为 ```self.batch_size = 1```

#### 5. default.yaml 修改:

原始为

save_model: False # Save the models to disk
save_model_interval: 50000 # Save models after this many timesteps

修改为

save_model: True # Save the models to disk
save_model_interval: 1000 # Save models after this many timesteps

#### 6. run.py 的改动（正式代码需要删除）
run.py line 237

新增一个指定500000变为2000 steps

#### 7. load actor/critic_init.th的改动
注意这里merge的policy是将子任务的参数平均，得到的共享参数，除非修改doe_controller，但那又涉及到role assignment的问题

#### 8. 固定长度one-hot的改动
ac.py line50, basic controller.py line89, doe_controller.py line115

#### TODO

run.py line 286 目前存储ckpt路径奇怪
save_path
'/data/qiaodan/projects/GRF_SUBTASK/doe_epymarl-main/results/models/ia2c_seed114514_scenario_layer2_decomposition0_subtask6_2025-05-05 11:45:34.649414/150'

line 215 目前并未成功load policy
因为ckpt path是 ""
而且需要解决team merged policy的问题，建议都存到decomposition/group文件夹下，但是分别命名为init policy和普通存储的final policy。
final policy用于在下一层target task训练时load，组合成init policy



doe的label和agent id要小心点，比如subtask 7 和 8 的合并顺序？
两个team都是0-N

注意 role list 作为MLP cls的train和load的label，检查这个label对超参数的影响


注意doe需要检查，预测的state对应label判断条件换成是否等于当前group id，因为train时候的label 替换成了group id为准，load doe时直接加载classifier nn params