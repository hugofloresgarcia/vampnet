# Scripts

## process_zip.py

Some requirements that may not be installed in the docker image:
* argbind
* wav2wav (pip install git+https://github.com/descriptinc/lyrebird-wav2wav.git or `pip install git+https://github.com/descriptinc/lyrebird-wav2wav.git@<branchname>`)

### zip folder structure

The zip folder should have the following internal structure:

```
base_folder/
    test_case_1/
        before.wav
    test_case_2/
        before.wav
    ...
    test_case_n/
        before.wav
```

Note: There can be issues with the output zip if the input zip folder structure is too deep or too shallow. IF you want/need to use a zip file with a different folder structure, adjust this:
https://github.com/descriptinc/lyrebird-wav2wav/blob/136c923ce19df03876a515ca0ed83854710cfa30/scripts/utils/process_zip.py#L28

### Execution
`python process_zip.py <path/to/zip> -tag <string>`
