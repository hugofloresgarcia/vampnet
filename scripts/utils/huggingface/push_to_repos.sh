# the (remote repo, model_name) are:
#  vampnet-music (default)
#  vampnet-percussion (percussion)
#  vampnet-choir ()'choir')
#  etc for..
# 'machines'
# 'n64'
#  'opera'
#   'percussion'

# iterate through remote, model_name pairs:
# and edit the DEFAULT_MODEL file in the repo
# add commit and push to the right remote
# each remote starts with https://huggingface.co/hugggof/{repo_name}

for repo in vampnet-music vampnet-percussion vampnet-choir vampnet-machines vampnet-n64 vampnet-opera vampnet-percussion
do
    echo "repo: $repo"
    # get the model name from the repo
    model_name=$(echo $repo | cut -d'-' -f2)
    # if the model_name is music , set it to default
    if [ $model_name == "music" ]; then
        model_name="default"
    fi
    echo "model_name: $model_name"
    # remove the DEFAULT_MODEL file
    rm DEFAULT_MODEL
    # create a new DEFAULT_MODEL file with the model name
    echo $model_name > DEFAULT_MODEL

    # commit and push to the right remote
    git add DEFAULT_MODEL
    git commit -m "update DEFAULT_MODEL to $model_name"
    git remote remove $repo
    git remote add $repo https://huggingface.co/spaces/hugggof/$repo
    git push $repo main
done