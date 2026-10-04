# Piano Skill Library

钢琴技能库的结构版，沿用 physicalRSI 的版本化能力契约、Memory / Tool / Skill 组合及证据继承方式。没有视频预览，不启动仿真，不注册或替换现有控制器。

## 目录

```text
piano_skill_library/
├── catalog.json                    # 技能目录、分类、状态与卡片路径
├── contracts.json                  # 输入输出、单位、坐标系、作用范围、评估要求
├── schemas/skill.schema.json        # 技能卡 JSON Schema
├── skills/
│   ├── primitives/                 # 单键触键释放、闲置手指回收
│   ├── coordination/               # 双手分配、邻键避碰、小位移选择
│   ├── contextual_repairs/         # 特定入口状态下的触点修复
│   ├── articulation/               # 持键、连奏与换指
│   ├── musicality/                 # 声部平衡、强弱与乐句
│   ├── composition/               # 保留状态的长时序组合
│   └── learning/                  # 失败记忆驱动的练习
├── memory/index.json               # 固定快照引用、记忆类型及检索字段设计
└── compositions/short-to-long.json # 短片段练习到全曲验证的组合结构
```

## 技能卡的含义

每张卡包含身份与结构版本、输入输出契约、作用范围、依赖技能、实现引用、参数、入口与出口条件、memory 读写关系、验证证据及组合约束。`structure-v1` 是结构版本，不是控制器内容摘要；真正执行时还需绑定实现、参数、模型、memory 和评估协议的内容版本。

- `existing_component`：仓库已有对应程序组件，不代表通用演奏资格。
- `experimental`：已有实验方法或局部修复，不能自动晋升为通用技能。
- `planned`：仅有结构占位；连奏换指、声部平衡和乐句强弱尚未实现为独立合格技能。

技能卡不是 Codex 的 SKILL.md，也不是新增训练完成的神经网络策略。旧技能卡的历史资格范围保留在证据引用中；这里不授予新的资格。

## 与现有代码的关系

现有运行时仍使用 `robopianist/rsi/skills.py`、`memory.py`、`continual.py` 和 `controller_schedule.py`。本目录提供面向钢琴的分类及统一结构；现有类还没有实现这里全部的契约、检索字段或准入条件。未来适配器应显式转换，不能直接把本目录 JSON 当作当前控制器配置。

`catalog.json` 内的卡片路径相对本目录；卡片中的代码及证据路径相对仓库根目录。`memory/index.json` 固定引用创建结构时的不可变快照，不自动跟随活动状态变化。历史诊断数据不强制符合新设计的标准化字段。

## 递归学习与组合

读取指定 memory 快照 → 选择失败窗口 → 从同一物理入口比较 incumbent 与候选 → 检查恢复段 → 连续全曲评估 → 保存证据 → 通过后按预期父版本继承。新修复必须从已经组合后的实际状态再练习。

串接需要保留完整积分状态与 MIDI 发声历史；两段都单独正确不保证组合正确。共享同一钢琴的动作不能并行写入，独立候选仿真可以隔离执行。跨曲目、移调和变速的复用必须分别验证；目前不声称已经通过。

## 评估范围

逐帧琴键 F1、起音 F1、错音、漏音、重复起音、持键时长、力度和手腕位移分开记录。仅减少错音但丢失原本正确音符的候选不应成为正式替代。声部与旋律需要明确标注，不能默认最高音就是旋律。

这些是库的目标契约；现有自动筛选尚未覆盖所有要求。软件结构校验只证明引用与格式一致，不证明物理演奏或音乐表现合格。

参考结构：`../PhysicalRSI_preview/docs/architecture.md` 和 `PhysicalRSI/Embodied_RSI/skills/{composition,routing}.py`。本目录未复制其运行时，也不引入跨仓库运行依赖。
