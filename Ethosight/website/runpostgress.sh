if [ $(docker ps -a -f name=djangopostgres | grep -w djangopostgres | wc -l) -eq 1 ]; then
    docker start djangopostgres
else
    docker run --name djangopostgres -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d postgres
fi
