python -m vampnet.db.init --config config/prosound.py
python -m vampnet.db.create --config config/prosound.py
python -m vampnet.db.preprocess --config config/prosound.py
python -m vampnet.db.partition --config config/prosound.py
python -m vampnet.db.train --config config/prosound.py