class DACDataset(torch.utils.data.Dataset):

    def __init__(self, 
        metadata_csvs: List[str] = None,
        seq_len: int = 512,
        split: str = None, 
        classlist: list = None,
        label_key: str = "label",
        class_weights: dict = None, 
        length: int = 1000000000
    ):
        assert metadata_csvs is not None, "Must provide metadata_csvs"

        self.seq_len = seq_len
        self.length = length
        
        # load the metadata csvs
        self.metadata = []
        for csv in metadata_csvs:
            self.metadata.append(pd.read_csv(csv))
        
        self.metadata = pd.concat(self.metadata)
        print(f"loaded metadata with {len(self.metadata.index)} rows")

        # filter by split
        if split is not None:
            self.metadata = filter_by_split(self.metadata, split)
        print(f"resolved split: {split}")
        print(f"metadata now has {len(self.metadata.index)} rows")

        # drop nans for all our path keys
        path_keys = ["dac_path"] # TODO: this is where conditioning paths would go too
        self.metadata = drop_nan(self.metadata, path_keys)
        print(f"dropped nans for missing paths in {path_keys}")
        print(f"metadata now has {len(self.metadata.index)} rows")

        # add roots to paths (to make all paths absolute)
        root_keys = [self.get_root_key(path_key) for path_key in path_keys]
        self.metadata = add_roots_to_paths(self.metadata, path_keys, root_keys)
        self.path_keys = path_keys

        # add p column for weighted sampling
        self.metadata = add_p_column(self.metadata)

        self.label_key = None
        if self.label_key is not None:
            assert label_key in self.metadata.columns, f"Class key {label_key} not in metadata columns {self.metadata.columns}"
            self.label_key = label_key
            
            # resolve classlist
            self.classlist = classlist if classlist is not None else self.metadata[label_key].unique().tolist()
            self.metadata = filter_by_classlist(self.metadata, classlist, label_key)
            print(f"resolved classlist: {self.classlist}")
            print(f'metadata now has {len(self.metadata.index)} rows')

            # resolve class weights
            self.class_weights = class_weights
            if self.class_weights is not None:
                self.metadata = apply_class_weights(self.metadata, class_weights, label_key)

    @property
    def input_key(self):
        return self.type_keys[0]

    @property
    def output_key(self):
        return self.type_keys[0]

    def __len__(self):
        return len(self.metadata)
    
    def get_root_key(self, path_key):
        return path_key.replace("_path", "_root")
    
    def __getitem__(self, idx, attempt=0):        
        smpld = self.metadata.sample(1, weights=self.metadata['p'])

        path_keys = list(self.path_keys)
        path_keys.pop(path_keys.index("dac_path"))

        # load our DAC data to start with 
        data = {}
        try:
            artifact = DACArtifact.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise e

        codes = artifact.codes

        nb, nc, nt = codes.shape
        if _batch_idx is None:
            # grab a random batch of codes
            _batch_idx = torch.randint(0, nb, (1,)).item()
        
        codes = codes[_batch_idx, :, :]

        if _start_idx is None:
            # get a seq_len chunk out of it
            if nt <= self.seq_len:
                _start_idx = 0
            else:
                _start_idx = torch.randint(0, nt - self.seq_len, (1,)).item()
        else:
            assert _start_idx + self.seq_len <= nt, f"start_idx {_start_idx} + seq_len {self.seq_len} must be less than nt {nt}. If you are using a paired dataset, is your input and output seq_len the same?"

        codes = codes[:, _start_idx:_start_idx + self.seq_len]

        codes, pad_mask = pad_if_needed(codes, self.seq_len)
        data["codes"] = codes
        data["ctx_mask"] = pad_mask
        data["start_idx"] = _start_idx
        data["batch_idx"] = _batch_idx

        for path_key in path_keys:
            key = path_key.replace("_path", "")
            try:
                path = smpld[f"{key}_path"].tolist()[0]
                # TODO: need to load the ConditionFeatures, 
                # then figure align with the codes by using the start_idx
                # and batch idx, then match the window sizes. 
                raise NotImplementedError(f"key {key} not implemented")

            except Exception as e:
                print(f"Error loading {idx}: {smpld}")
                raise e
                
        # grab the labels
        if self.label_key is not None:
            label = smpld[self.label_key].tolist()[0]
            label = torch.tensor(self.classlist.index(label), dtype=torch.long)
        else:
            label = None
        data["label"] = label
            
        return data