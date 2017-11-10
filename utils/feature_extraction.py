#coding=utf-8
import os
import numpy as np
import datetime
from enum import Enum
from general_utils import get_pickle, dump_pickle, get_vocab_dict

NULL = "<null>"
UNK = "<unk>"
ROOT = "<root>"
pos_prefix = "<p>:"
dep_prefix = "<d>:"
punc_pos = ["''", "``", ":", ".", ","]

today_date = str(datetime.datetime.now().date())


class DataConfig:  # data, embedding, model path etc.
    # Data Paths
    data_dir_path = "./data"
    train_path = "tt.conll"#"train.conll"
    valid_path = "dd.conll"#"dev.conll"
    test_path = "te.conll"#"test.conll"

    # embedding
    embedding_file = "en-cw.txt"

    # model saver
    model_dir = "params_" + today_date
    model_name = "parser.weights"

    # summary
    summary_dir = "params_" + today_date
    train_summ_dir = "train_summaries"
    test_summ_dir = "valid_summaries"

    # dump - vocab
    dump_dir = "./data/dump"
    word_vocab_file = "word2idx.pkl"
    pos_vocab_file = "pos2idx.pkl"
    dep_vocab_file = "dep2idx.pkl"

    # dump - embedding
    word_emb_file = "word_emb.pkl"  # 2d array
    pos_emb_file = "pos_emb.pkl"  # 2d array
    dep_emb_file = "dep_emb.pkl"  # 2d array

# 神经网络配置
class ModelConfig(object):  # Takes care of shape, dimensions used for tf model
    # Input
    word_features_types = None
    pos_features_types = None
    dep_features_types = None
    num_features_types = None
    embedding_dim = 50

    # hidden_size
    l1_hidden_size = 200
    l2_hidden_size = 15

    # output
    num_classes = 3

    # Vocab
    word_vocab_size = None
    pos_vocab_size = None
    dep_vocab_size = None

    # num_epochs
    n_epochs = 20

    # batch_size
    batch_size = 2048

    # dropout
    keep_prob = 0.5
    reg_val = 1e-8

    # learning_rate
    lr = 0.001

    # load existing vocab
    load_existing_vocab = False

    # summary
    write_summary_after_epochs = 1

    # valid run
    run_valid_after_epochs = 1


class SettingsConfig:  # enabling and disabling features, feature types
    # Features
    use_word = True
    use_pos = True
    use_dep = True
    is_lower = True


class Flags(Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3


class Token(object):
    def __init__(self, token_id, word, pos, dep, head_id):
        self.token_id = token_id  # 单词在句子中的索引
        self.word = word.lower() if SettingsConfig.is_lower else word # 单词为英文时 大写转换成小写
        self.pos = pos_prefix + pos # '<p>:NNP'  词性标签
        self.dep = dep_prefix + dep # '<d>:compound' 依赖
        self.head_id = head_id  # 父依赖的ID head token index
        self.predicted_head_id = None # 预测的父依赖
        self.left_children = list() # 左孩子
        self.right_children = list() # 右孩子


    def is_root_token(self):
        if self.word == ROOT:
            return True
        return False


    def is_null_token(self):
        if self.word == NULL:
            return True
        return False


    def is_unk_token(self):
        if self.word == UNK:
            return True
        return False


    def reset_predicted_head_id(self):
        self.predicted_head_id = None


NULL_TOKEN = Token(-1, NULL, NULL, NULL, -1)
ROOT_TOKEN = Token(-1, ROOT, ROOT, ROOT, -1)
UNK_TOKEN = Token(-1, UNK, UNK, UNK, -1)


class Sentence(object):
    def __init__(self, tokens):
        self.Root = Token(-1, ROOT, ROOT, ROOT, -1) #根节点
        self.tokens = tokens #非根节点其它节点
        self.buff = [token for token in self.tokens] #非根节点其它节点 对self.tokens深层复制
        self.stack = [self.Root] #栈
        self.dependencies = []
        self.predicted_dependencies = []


    def load_gold_dependency_mapping(self):
        for token in self.tokens:
            if token.head_id != -1:
                token.parent = self.tokens[token.head_id]
                if token.head_id > token.token_id:
                    token.parent.left_children.append(token.token_id)
                else:
                    token.parent.right_children.append(token.token_id)
            else:
                token.parent = self.Root

        for token in self.tokens:
            token.left_children.sort()
            token.right_children.sort()

    # 跟新父元素的左右孩子数组元素
    def update_child_dependencies(self, curr_transition):
        if curr_transition == 0: #left arc
            head = self.stack[-1]  # 获取父元素
            dependent = self.stack[-2] # 获取子元素
        elif curr_transition == 1:
            head = self.stack[-2]
            dependent = self.stack[-1]

        if head.token_id > dependent.token_id: #left arc 子元素在父元素的左变
            head.left_children.append(dependent.token_id) # 把左边的子元素的ID添加到父元素的左孩子数组中
            head.left_children.sort()
        else:
            head.right_children.append(dependent.token_id)
            head.right_children.sort()
            # dependent.head_id = head.token_id

    # token ： sentence.stack[- 1]
    def get_child_by_index_and_depth(self, token, index, direction, depth):  # Get child token
        if depth == 0:
            return token
        # 开始时，跟节点没有 左右孩子
        if direction == "left":
            if len(token.left_children) > index: # 有左孩子 深度遍历获取depth = 1 左孩子  depth = 2 左孩子的左孩子
                return self.get_child_by_index_and_depth(
                    self.tokens[token.left_children[index]], index, direction, depth - 1)
            return NULL_TOKEN
        else: # 有左孩子 深度遍历获取 depth = 1 右孩子的右孩子 depth = 2 右孩子的右孩子
            if len(token.right_children) > index:
                return self.get_child_by_index_and_depth(
                    self.tokens[token.right_children[::-1][index]], index, direction, depth - 1)
            return NULL_TOKEN

    # len(self.stack) = 1 labels = [0,0,1]  len(self.stack) = 2 labels = [0,1,1] len(self.stack) >= 3 labels = [1,1,1]
    def get_legal_labels(self):
        labels = ([1] if len(self.stack) > 2 else [0])
        labels += ([1] if len(self.stack) >= 2 else [0])
        labels += [1] if len(self.buff) > 0 else [0]
        return labels


    def get_transition_from_current_state(self):  # logic to get next transition
        if len(self.stack) < 2:
            return 2  # shift
        # 当len(self.stack) > =2 获取最后两个元素
        stack_token_0 = self.stack[-1] # 最后一个元素
        stack_token_1 = self.stack[-2] # 倒数第二个元素
        if stack_token_1.token_id >= 0 and stack_token_1.head_id == stack_token_0.token_id:  # left arc
            return 0 #倒数第二元素不是root节点 & 并且倒数第二个节点的父节点时最后一个节点，即：stack_token_1 <--- stack_token_0
        elif stack_token_1.token_id >= -1 and stack_token_0.head_id == stack_token_1.token_id \
                and stack_token_0.token_id not in map(lambda x: x.head_id, self.buff):
            return 1  # right arc # stack_token_1 ---> stack_token_0 & 最后一个元素不存在buf的缓存中
        else: # stack_token_1 和 stack_token_0 不存在任何关系
            return 2 if len(self.buff) != 0 else None

    # 更新句子中的转移关系 或者预测出的依赖关系
    def update_state_by_transition(self, transition, gold=True):  # updates stack, buffer and dependencies
        if transition is not None:
            if transition == 2:  # shift
                self.stack.append(self.buff[0])
                self.buff = self.buff[1:] if len(self.buff) > 1 else []
            elif transition == 0:  # left arc   self.stack[-2]  <----- self.stack[-1]
                self.dependencies.append( # 训练时gold = true 执行elf.dependencies.append  预测时gold = ：self.predicted_dependencies.append
                    (self.stack[-1], self.stack[-2])) if gold else self.predicted_dependencies.append(
                    (self.stack[-1], self.stack[-2])) # （父元素，子元素）
                self.stack = self.stack[:-2] + self.stack[-1:] # 删除stack栈中倒数第二个元素
            elif transition == 1:  # right arc
                self.dependencies.append(
                    (self.stack[-2], self.stack[-1])) if gold else self.predicted_dependencies.append(
                    (self.stack[-2], self.stack[-1]))
                self.stack = self.stack[:-1]


    def reset_to_initial_state(self):
        self.buff = [token for token in self.tokens]
        self.stack = [self.Root]


    def clear_prediction_dependencies(self):
        self.predicted_dependencies = []


    def clear_children_info(self):
        for token in self.tokens:
            token.left_children = []
            token.right_children = []

# model_config 神经网络配置参数
# feature_extractor
# 设置参数数据集
class Dataset(object):
    def __init__(self, model_config, train_data, valid_data, test_data, feature_extractor):
        self.model_config = model_config
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.feature_extractor = feature_extractor

        # Vocab
        self.word2idx = None # 单词-》index
        self.idx2word = None # index -> 单词
        self.pos2idx = None
        self.idx2pos = None
        self.dep2idx = None
        self.idx2dep = None

        # Embedding Matrix
        self.word_embedding_matrix = None
        self.pos_embedding_matrix = None
        self.dep_embedding_matrix = None

        # input & outputs
        self.train_inputs, self.train_targets = None, None
        self.valid_inputs, self.valid_targets = None, None
        self.test_inputs, self.test_targets = None, None

    # 当字典不存在时候，读取训练集并生成单词、词性和依赖字典
    def build_vocab(self):

        all_words = set()
        all_pos = set()
        all_dep = set()

        for sentence in self.train_data:
            all_words.update(set(map(lambda x: x.word, sentence.tokens)))
            all_pos.update(set(map(lambda x: x.pos, sentence.tokens)))
            all_dep.update(set(map(lambda x: x.dep, sentence.tokens)))

        all_words.add(ROOT_TOKEN.word) # "<root>"
        all_words.add(NULL_TOKEN.word) # "<null>"
        all_words.add(UNK_TOKEN.word) # "<unk>"

        all_pos.add(ROOT_TOKEN.pos) # "<root>"
        all_pos.add(NULL_TOKEN.pos) # "<null>"
        all_pos.add(UNK_TOKEN.pos) # "<unk>"

        all_dep.add(ROOT_TOKEN.dep) # "<root>"
        all_dep.add(NULL_TOKEN.dep) # "<null>"
        all_dep.add(UNK_TOKEN.dep) # "<unk>"

        word_vocab = list(all_words)
        pos_vocab = list(all_pos)
        dep_vocab = list(all_dep)

        word2idx = get_vocab_dict(word_vocab)
        idx2word = {idx: word for (word, idx) in word2idx.items()}

        pos2idx = get_vocab_dict(pos_vocab)
        idx2pos = {idx: pos for (pos, idx) in pos2idx.items()}

        dep2idx = get_vocab_dict(dep_vocab)
        idx2dep = {idx: dep for (dep, idx) in dep2idx.items()}

        self.word2idx = word2idx
        self.idx2word = idx2word

        self.pos2idx = pos2idx
        self.idx2pos = idx2pos

        self.dep2idx = dep2idx
        self.idx2dep = idx2dep

    # 把训练集单词 词性 和 依赖 转行成对应的词向量
    def build_embedding_matrix(self):

        # load word vectors
        word_vectors = {}
        embedding_lines = open(os.path.join(DataConfig.data_dir_path, DataConfig.embedding_file), "r").readlines() # ./data/en-cw.txt
        for line in embedding_lines:
            sp = line.strip().split()
            word_vectors[sp[0]] = [float(x) for x in sp[1:]]

        # word embedding
        self.model_config.word_vocab_size = len(self.word2idx)
        word_embedding_matrix = np.asarray(
            np.random.normal(0, 0.9, size=(self.model_config.word_vocab_size, self.model_config.embedding_dim)),
            dtype=np.float32) # [51,50] 赋根据正太分布的随机值
        for (word, idx) in self.word2idx.items():
            if word in word_vectors:
                word_embedding_matrix[idx] = word_vectors[word]
            elif word.lower() in word_vectors:
                word_embedding_matrix[idx] = word_vectors[word.lower()]
        self.word_embedding_matrix = word_embedding_matrix

        # 词性和依赖时根据正太分布的随机值生成50位的数据
        # -------------------------------------------------------------------------------------------------------- #
        # pos embedding
        self.model_config.pos_vocab_size = len(self.pos2idx)
        pos_embedding_matrix = np.asarray(
            np.random.normal(0, 0.9, size=(self.model_config.pos_vocab_size, self.model_config.embedding_dim)),
            dtype=np.float32)
        self.pos_embedding_matrix = pos_embedding_matrix

        # dep embedding
        self.model_config.dep_vocab_size = len(self.dep2idx)
        dep_embedding_matrix = np.asarray(
            np.random.normal(0, 0.9, size=(self.model_config.dep_vocab_size, self.model_config.embedding_dim)),
            dtype=np.float32)
        self.dep_embedding_matrix = dep_embedding_matrix
        # -------------------------------------------------------------------------------------------------------- #


    def convert_data_to_ids(self):
        self.train_inputs, self.train_targets = self.feature_extractor. \
            create_instances_for_data(self.train_data, self.word2idx, self.pos2idx, self.dep2idx)

        # self.valid_inputs, self.valid_targets = self.feature_extractor.\
        #     create_instances_for_data(self.valid_data, self.word2idx)
        # self.test_inputs, self.test_targets = self.feature_extractor.\
        #     create_instances_for_data(self.test_data, self.word2idx)


    def add_to_vocab(self, words, prefix=""):
        idx = len(self.word2idx)
        for token in words:
            if prefix + token not in self.word2idx:
                self.word2idx[prefix + token] = idx
                self.idx2word[idx] = prefix + token
                idx += 1


class FeatureExtractor(object):
    def __init__(self, model_config):
        self.model_config = model_config

    # len(tokens) = 6 tokens【前三个为【stack元素后三个元素】，后三个为【buff的前三个元素】】提取stack栈里面的最后面的三个元素，如果不足三个，前面用null节点填补，
    def extract_from_stack_and_buffer(self, sentence, num_words=3):
        tokens = []
        # --------------------- 加入三个空节点 ------------------------------------------ #
        # len(sentence.stack)) = 1 时 【【null】,[null]】
        tokens.extend([NULL_TOKEN for _ in range(num_words - len(sentence.stack))]) # sentence.stack = ROOT节点
        # len(sentence.stack)) <= 3 追加所有sentence.stack元素  否则 追加sentence.stack 后三个元素
        tokens.extend(sentence.stack[-num_words:])
        # --------------------- 加入三个空节点 ------------------------------------------ #
        # 在三个空节点后在后面  加入 句子的前三个单词节点
        # [null.null,root,word1,word2,word3]
        tokens.extend(sentence.buff[:num_words])
        tokens.extend([NULL_TOKEN for _ in range(num_words - len(sentence.buff))]) # len(sentence.buff) >= num_words 时 不执行该句
        return tokens  # 6 features

    # 获取【第一个左右孩子】 【第二个左孩子  第二个右孩子】 【第一个左孩子的左孩子，第一个右孩子的右孩子】
    def extract_children_from_stack(self, sentence, num_stack_words=2):
        children_tokens = []
        # 遍历后两个元素 ，先遍历最后一个元素，再遍历倒数第二个元素
        for i in range(num_stack_words): # num_stack_words = 2
            if len(sentence.stack) > i: # len(sentence.stack) 开始时 = 1
                lc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "left", 1) # sentence.stack[-i - 1] 获取stack[] 最前一个元素 栈后进先出
                rc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "right", 1)
                # lc0 = NULL 不执行 lc1
                lc1 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 1, "left",
                                                            1) if lc0 != NULL_TOKEN else NULL_TOKEN
                # rc0 = NULL 不执行rc1
                rc1 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 1, "right",
                                                            1) if rc0 != NULL_TOKEN else NULL_TOKEN
                # lc0 = NULL 不执行 llc0
                llc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "left",
                                                             2) if lc0 != NULL_TOKEN else NULL_TOKEN
                # rc0 = NULL 不执行 rrc0
                rrc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "right",
                                                             2) if rc0 != NULL_TOKEN else NULL_TOKEN

                children_tokens.extend([lc0, rc0, lc1, rc1, llc0, rrc0])#[【第一个左右孩子】 【第二个左孩子】  【第二个右孩子】 【第一个左孩子的左孩子】【第一个右孩子的右孩子】 ]
            else:
                [children_tokens.append(NULL_TOKEN) for _ in range(6)]

        return children_tokens  # 12 features


    def extract_for_current_state(self, sentence, word2idx, pos2idx, dep2idx):
        # 0 : [null.null,root,word1,word2,word3]
        direct_tokens = self.extract_from_stack_and_buffer(sentence, num_words=3)
        # [[null] * 12]
        children_tokens = self.extract_children_from_stack(sentence, num_stack_words=2)

        word_features = []
        pos_features = []
        dep_features = []

        # Word features -> 18
        word_features.extend(map(lambda x: x.word, direct_tokens))
        word_features.extend(map(lambda x: x.word, children_tokens))

        # pos features -> 18
        # 0: [null.null,root,word1,word2,word3] 1: [null,root,word1,word2,word3,word4]
        pos_features.extend(map(lambda x: x.pos, direct_tokens))
        # [null.null,null,word1,word2,word3,[null] * 12]
        pos_features.extend(map(lambda x: x.pos, children_tokens))

        # dep features -> 12 (only children)
        # [[null] * 12]
        dep_features.extend(map(lambda x: x.dep, children_tokens))
        # 把单词、词性和依赖通过字典转换成对应的索引index
        word_input_ids = [word2idx[word] if word in word2idx else word2idx[UNK_TOKEN.word] for word in word_features] # UNK_TOKEN.word = <unk>
        pos_input_ids = [pos2idx[pos] if pos in pos2idx else pos2idx[UNK_TOKEN.pos] for pos in pos_features]
        dep_input_ids = [dep2idx[dep] if dep in dep2idx else dep2idx[UNK_TOKEN.dep] for dep in dep_features]

        return [word_input_ids, pos_input_ids, dep_input_ids]  # 48 features


    def create_instances_for_data(self, data, word2idx, pos2idx, dep2idx):
        lables = []
        word_inputs = []
        pos_inputs = []
        dep_inputs = []
        for i, sentence in enumerate(data):
            num_words = len(sentence.tokens)

            for _ in range(num_words * 2):
                if _ == 6:
                    print _
                word_input, pos_input, dep_input = self.extract_for_current_state(sentence, word2idx, pos2idx, dep2idx)
                legal_labels = sentence.get_legal_labels() # len(sentence.stack) = 1 labels = [0,0,1]  len(sentence.stack) = 2 labels = [0,1,1] len(sentence.stack) >= 3 labels = [1,1,1]
                # transition == 2 # shift
                # transition == 0:  # left arc
                # transition == 1:  # right arc
                curr_transition = sentence.get_transition_from_current_state() # 2
                if curr_transition is None:
                    break
                assert legal_labels[curr_transition] == 1

                # Update left/right children
                if curr_transition != 2:
                    sentence.update_child_dependencies(curr_transition)
                # 通过sentence 执行 shift、left arc和right arc三个状态其一
                # stack[]   buff[]
                sentence.update_state_by_transition(curr_transition) # # 更新句子中的转移关系 或者预测出的依赖关系
                lables.append(curr_transition)  # 存储上一个执行的状态
                word_inputs.append(word_input)
                pos_inputs.append(pos_input)
                dep_inputs.append(dep_input)

            else:
                sentence.reset_to_initial_state() # 恢复stack[root] 和 buff[tokens] 的数据

            # reset stack and buffer to default state
            sentence.reset_to_initial_state()
        # lables 句子转移动作集 self.model_config.num_classes = 0 output len(labels) = 110 tagets[110][3]
        targets = np.zeros((len(lables), self.model_config.num_classes), dtype=np.int32)
        targets[np.arange(len(targets)), lables] = 1 # lables 一维 转换成三维 如： 0 -->【1,0,0】 1 -->【0,1,0】 2 -->【0,0,1】

        return [word_inputs, pos_inputs, dep_inputs], targets


class DataReader(object):
    def __init__(self):
        print "A"


    def read_conll(self, token_lines):
        tokens = []
        for each in token_lines:
            fields = each.strip().split("\t")
            token_index = int(fields[0]) - 1
            word = fields[1]
            pos = fields[4]
            dep = fields[7]
            head_index = int(fields[6]) - 1
            token = Token(token_index, word, pos, dep, head_index)
            tokens.append(token)
        sentence = Sentence(tokens)

        # sentence.load_gold_dependency_mapping()
        return sentence


    def read_data(self, data_lines):
        data_objects = []
        token_lines = []
        for token_conll in data_lines:
            token_conll = token_conll.strip()
            if len(token_conll) > 0:
                token_lines.append(token_conll)
            else:
                data_objects.append(self.read_conll(token_lines)) # data_objects 存储句子的数字
                token_lines = []
        if len(token_lines) > 0:
            data_objects.append(self.read_conll(token_lines))
        return data_objects


def load_datasets(load_existing_dump=False):
    model_config = ModelConfig()

    data_reader = DataReader()
    train_lines = open(os.path.join(DataConfig.data_dir_path, DataConfig.train_path), "r").readlines() # train.conll
    valid_lines = open(os.path.join(DataConfig.data_dir_path, DataConfig.valid_path), "r").readlines() # dev.conll
    test_lines = open(os.path.join(DataConfig.data_dir_path, DataConfig.test_path), "r").readlines() # test.conll

    # Load data
    train_data = data_reader.read_data(train_lines) #[sent[token[]]...]
    print ("Loaded Train data")
    valid_data = data_reader.read_data(valid_lines)
    print ("Loaded Dev data")
    test_data = data_reader.read_data(test_lines)
    print ("Loaded Test data")

    feature_extractor = FeatureExtractor(model_config)
    dataset = Dataset(model_config, train_data, valid_data, test_data, feature_extractor)

    # Vocab processing
    if load_existing_dump: # 单词字典 词性字典 依赖字典 文件存在时，读取本地文件
        # 单词字典
        dataset.word2idx = get_pickle(os.path.join(DataConfig.dump_dir, DataConfig.word_vocab_file)) # './data/dump/word2idx.pkl'
        dataset.idx2word = {idx: word for (word, idx) in dataset.word2idx.items()}
        # 词性字典
        dataset.pos2idx = get_pickle(os.path.join(DataConfig.dump_dir, DataConfig.pos_vocab_file)) # pos2idx.pkl
        dataset.idx2pos = {idx: pos for (pos, idx) in dataset.pos2idx.items()}
        # 依赖字典
        dataset.dep2idx = get_pickle(os.path.join(DataConfig.dump_dir, DataConfig.dep_vocab_file))
        dataset.idx2dep = {idx: dep for (dep, idx) in dataset.dep2idx.items()}

        dataset.model_config.load_existing_vocab = True
        print "loaded existing Vocab!"
        # 读取本地词向量（单词\词性\依赖）
        dataset.word_embedding_matrix = get_pickle(os.path.join(DataConfig.dump_dir, DataConfig.word_emb_file))
        dataset.pos_embedding_matrix = get_pickle(os.path.join(DataConfig.dump_dir, DataConfig.pos_emb_file))
        dataset.dep_embedding_matrix = get_pickle(os.path.join(DataConfig.dump_dir, DataConfig.dep_emb_file))
        print "loaded existing embedding matrix!"

    else: # 当数据集不存在时
        dataset.build_vocab() # 读取训练集中的单词、词性和依赖
        dump_pickle(dataset.word2idx, os.path.join(DataConfig.dump_dir, DataConfig.word_vocab_file)) # './data/dump/word2idx.pkl'
        dump_pickle(dataset.pos2idx, os.path.join(DataConfig.dump_dir, DataConfig.pos_vocab_file)) # pos2idx.pkl
        dump_pickle(dataset.dep2idx, os.path.join(DataConfig.dump_dir, DataConfig.dep_vocab_file)) # dep2idx.pkl
        dataset.model_config.load_existing_vocab = True
        print "Vocab Build Done!"
        # 把训练集单词 词性 和 依赖 转行成对应的词向量
        dataset.build_embedding_matrix()
        print "embedding matrix Build Done"
        # 存储词向量
        dump_pickle(dataset.word_embedding_matrix, os.path.join(DataConfig.dump_dir, DataConfig.word_emb_file)) # ./data/dump/word_emb.pkl
        dump_pickle(dataset.pos_embedding_matrix, os.path.join(DataConfig.dump_dir, DataConfig.pos_emb_file)) # ./data/dump/pos_emb.pkl
        dump_pickle(dataset.dep_embedding_matrix, os.path.join(DataConfig.dump_dir, DataConfig.dep_emb_file)) # # ./data/dump/dep_emb.pkl

    print "converting data into ids.."
    dataset.convert_data_to_ids() # inputs(train_inputs) [词索引，词性索引，依赖索引]  output(train_targets) [转移状态]
    print "Done!"
    dataset.model_config.word_features_types = len(dataset.train_inputs[0][0]) # 单词的个数
    dataset.model_config.pos_features_types = len(dataset.train_inputs[1][0]) # 词性的个数
    dataset.model_config.dep_features_types = len(dataset.train_inputs[2][0]) # 依赖的个数
    dataset.model_config.num_features_types = dataset.model_config.word_features_types + \
                                              dataset.model_config.pos_features_types + dataset.model_config.dep_features_types # 单词的个数 + 词性的个数 + 依赖的个数
    dataset.model_config.num_classes = len(dataset.train_targets[0]) # 转移状态 的维数

    return dataset
