docker exec -it $(docker container ls  | grep 'pro_gan_tf' | awk '{print $12}') /bin/bash
