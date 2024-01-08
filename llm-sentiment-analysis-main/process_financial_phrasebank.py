import random

out = []
with open('data/financial_phrasebank.csv', 'r', encoding='utf-8', errors='ignore') as fp:
    lines = fp.read().strip().split('\n')
    for l in lines:
        if l.startswith('positive,'):
            label = 'positive'
            text = l[len('positive,'):]
        elif l.startswith('neutral,'):
            label = 'neutral'
            text = l[len('neutral,'):]
        elif l.startswith('negative,'):
            label = 'negative'
            text = l[len('negative,'):]
        else:
            raise ValueError(l)
        text = text.replace('``', '')
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' %', '%')
        text = text.replace(" 's", "'s")
        text = text.replace(" ' ", "' ")
        text = text.replace(" 'm", "'m")
        text = text.strip().strip('"')
        # print('>>>', text, '\n')
        ll = f'{label}\t\t{text}'
        assert len(ll.split('\t\t')) == 2, l
        out.append(ll)

test_ratio = 0.2
val_ratio = 0.2
random.seed(123)
random.shuffle(out)
n_test = int(len(out)*test_ratio)
test_set = out[-n_test:]
train_set = out[:-n_test]
n_val = int(len(train_set)*val_ratio)
val_set = train_set[-n_val:]
train_set = train_set[:-n_val]

with open('data/train_data.txt', 'w+') as fp:
    fp.write('\n'.join(train_set))
with open('data/val_data.txt', 'w+') as fp:
    fp.write('\n'.join(val_set))
with open('data/test_data.txt', 'w+') as fp:
    fp.write('\n'.join(test_set))
print('saved to train_data.txt, val_data.txt and test_data.txt')
