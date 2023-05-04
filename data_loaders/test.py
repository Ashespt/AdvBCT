from data_loaders import build_train_dataloader,build_test_dataloader

print('test gldv2 gldv2_train_old_30percent_class.txt')
train_loader = build_train_dataloader('gldv2',root='./data/GLDv2',file_dir='./data/GLDv2/gldv2_train_old_30percent_class.txt')
for img,label in train_loader:
    print(img.shape)
    print(label)
    break
print('test ms1m ms1m_train_old_30percent_class.txt')
train_loader = build_train_dataloader('ms1m',root='./data/ms1m',file_dir='data/ms1m/ms1m_train_old_30percent_class.txt')
for img,label in train_loader:
    print(img.shape)
    print(label)
    break
print('test roxford5k')
test_loader,gts = build_test_dataloader('roxford5k',root='./data/ROxfordParis/')
for img,label in test_loader:
    print(img.shape)
    print(label)
    break
print('test rparis6k')
test_loader,gts = build_test_dataloader('rparis6k',root='./data/ROxfordParis/')
for img,label in test_loader:
    print(img.shape)
    print(label)
    break
