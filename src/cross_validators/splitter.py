import numpy as np

def stratified_sequential_split(ntest, labels):
    n = len(labels)
    ntrain = n - ntest
    unique_labels = np.unique(labels)

    ntrain_per_class = int(ntrain//len(unique_labels))
    ntest_per_class = int(ntest//len(unique_labels))

    train_idx = []
    test_idx = []
    for cls in unique_labels:
        cls_indices = np.where(labels == cls)[0]
        train_cls_indices = cls_indices[:ntrain_per_class]
        test_cls_indices = cls_indices[ntrain_per_class:ntrain_per_class+ntest_per_class]

        train_idx.extend(train_cls_indices)
        test_idx.extend(test_cls_indices)
    return train_idx, test_idx


def train_val(subjects_df, n_train_subjects, n_val_subjects = 4, seed = 8):
    import random
    random.seed(seed)
    all_subjects = subjects_df["subject_id"].tolist()
    train_subjects = random.sample(all_subjects, n_train_subjects)
    if n_val_subjects == 0 :
        val_subjects = []
    else:
        val_subjects = random.sample(list(set(all_subjects) - set(train_subjects)), n_val_subjects)
    ignored_participants = list(set(all_subjects) - set(val_subjects) - set(train_subjects))
    return train_subjects, val_subjects, ignored_participants

def train_val_balanceBySex(subjects_df,  n_train_subjects, n_val_subjects = 4, seed = 8, ):
    import random
    random.seed(seed)
    all_subjects = subjects_df["subject_id"].tolist()
    female_subjects = subjects_df[subjects_df['sex'] == 'F']["subject_id"].tolist()
    male_subjects = subjects_df[subjects_df['sex'] == 'M']["subject_id"].tolist()    
    # Obtain n_train_subjects // 2 subjects for train
    n_train_per_sex = n_train_subjects // 2

    random_female = random.sample(female_subjects, n_train_per_sex)
    random_male = random.sample(male_subjects, n_train_per_sex)
    train_subjects = random_female + random_male
    random.shuffle(train_subjects)
    female_subjects = list(set(female_subjects) - set(random_female))
    male_subjects = list(set(male_subjects) - set(random_male))

    if n_val_subjects == 0 :
        ignored_participants = list(set(all_subjects) - set(train_subjects))
        val_subjects = []
    else:
        # CHeck if n_val_subjects is divisible by 2
        if n_val_subjects % 2 != 0:
            raise ValueError("n_val_subjects must be divisible by 2")
        # Obtain n_val_subjects // 2 subjects for val
        n_val_per_sex = n_val_subjects // 2
        random_female = random.sample(female_subjects, n_val_per_sex)
        random_male = random.sample(male_subjects, n_val_per_sex)
        val_subjects = random_female + random_male
        random.shuffle(val_subjects)
        female_subjects = list(set(female_subjects) - set(random_female))
        male_subjects = list(set(male_subjects) - set(random_male))
        # Finding the subjects that were eliminated
        ignored_participants = list(set(all_subjects) - set(val_subjects) - set(train_subjects))
    return train_subjects, val_subjects, ignored_participants




