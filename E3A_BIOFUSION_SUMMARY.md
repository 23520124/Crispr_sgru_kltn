# E3a BioFusion va E3a Pairwise

## 1. Tac gia: `Train.py` + `MODEL.py`

- `Train.py` goi mo hinh `Crispr-SGRU` tu `MODEL.py`; dau vao la ma hoa cap sgRNA/off-target (shape 1 x 24 x 7).
- `MODEL.py` dung Inception Conv2D (kernel 1x1, 1x2, 1x3, 1x5), ba tang BiGRU (30/20/10), Flatten, Dense 128/64, Dropout 0.35 va dau ra 2 sigmoid. Loss trong ham nay la binary cross-entropy; Adam 1e-4.
- `Train.py` tach rieng 15% test truoc. Tu 85% con lai tach validation 20% (tong the xap xi 68% train / 17% val / 15% test). Vong lap 5 lan train tiep tren cung mot model, moi lan 30 epoch; khong khoi tao lai model giua cac lan. Sau 150 epoch lien tuc, model duoc danh gia mot lan tren test set co dinh.
- Generator cua tac gia tao batch can bang 1:1: tham so `batchsize=256` la 256 cap; moi batch thuc te gom 256 negative + 256 positive = 512 mau. Negative duoc quet theo thu tu da shuffle, positive lay lai co hoan lai. Validation moi lan lay mot batch 100 negative + 100 positive.

## 2. MODEL12 khac MODEL.py the nao?

- Backbone xu ly chuoi van cung ho Inception + BiGRU (30/20/10) + Dense; MODEL12 la mot bien the.
- Khac biet ro trong code dang co: `MODEL.py` co dau ra 2 sigmoid va Binary Cross-Entropy; `MODEL12.PY` co dau ra 1 sigmoid va BCE-Dice (0.5 BCE + 0.5 Soft Dice). MODEL12 cung la backbone ma `MODEL_E3_BioFusion.py` su dung cho nhanh chuoi.

## 3. E3a BioFusion khac Train.py/MODEL.py the nao?

- Them mot nhanh dac trung sinh hoc vao backbone MODEL12: E3a co 25 dac trung mismatch (tong mismatch, phan bo theo vung, mismatch mask theo vi tri); khong co PAM, GC hay indel. Nhanh sinh hoc duoc chuan hoa bang scaler fit tren train fold, sau do noi voi bieu dien chuoi truoc dau ra.
- Dung BCE-Dice, Adam 1e-4 va StableMining sampling: batch 256 gom 64 positive, 64 hard negative (mismatch <= 3), 128 normal negative; 120 steps/epoch, toi da 35 epoch, giam learning rate theo validation loss va luu checkpoint val_loss tot nhat.
- Protocol E3 tach 15% test mot lan truoc khi chia fold. 85% con lai duoc chia StratifiedKFold 5 fold: moi fold co validation rieng, nhung ca 5 model deu duoc danh gia tren cung test set 15%. Test khong duoc tach lai rieng trong moi fold. Moi fold khoi tao model moi; khong phai 5 dot train noi tiep nhu `Train.py`.

## 4. E3a Pairwise khac E3a the nao?

- Giu E3a features (25 mismatch features), kien truc BioFusion MODEL12 va protocol E3 (test 15% co dinh, 5 validation folds, 35 epoch/fold).
- Them ranking loss: `total = BCE-Dice + ranking_weight * pairwise_softplus`. Pairwise loss khuyen khich positive co diem cao hon negative gan nhat cung sgRNA.
- Thay doi sampling hard negative: thay 64 hard negative mismatch <=3 cua E3 bang 64 negative cung guide, chon tu toi da 32 mau gan positive theo signature mismatch vi tri/base; van giu 64 positive va 128 normal negative.
- Vi ca loss va cach lay hard negative cung doi, khong the quy moi thay doi ket qua cho ranking loss rieng. `ranking_weight=0` van dung sampler pairwise, nen khong phai E3 goc.
