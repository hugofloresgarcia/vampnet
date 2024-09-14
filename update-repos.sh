# 
repos=( "vampnet-music" "vampnet-percussion" "vampnet-n64" "vampnet-birds" "vampnet-choir" "vampnet-machines" "nesquik" "vampnet-opera")
for repo in "${repos[@]}"
do
    echo "Updating $repo"
    git remote add --fetch $repo https://huggingface.co/spaces/hugggof/$repo
    git push --force $repo main
done

# https://huggingface.co/spaces/hugggof/vampnet-music
# git push --space-percussion main 