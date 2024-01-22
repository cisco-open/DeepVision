#!/usr/bin/env python3
import click
import redis
import uuid

r = redis.Redis(host='localhost', port=6379, db=0)

@click.group()
def cli():
    """Access Code Management Tool"""
    pass

@click.command()
@click.option('--num', default=1000, help='Number of access codes to generate.')
def generateaccesscodes(num):
    """Generate and store access codes in Redis."""
    codes = [str(uuid.uuid4()) for _ in range(num)]
    
    click.echo(f"Generated {len(codes)} access codes.")
    
    count = 0
    for code in codes:
        response = r.hset("access_codes", code, "unused")
        if response == 1:
            count += 1
            
    click.echo(f"Stored {count} new access codes in Redis.")

@click.command()
def listaccesscodes():
    """List all access codes and their statuses, sorted by unused first."""
    codes = r.hgetall("access_codes")
    sorted_codes = sorted(codes.items(), key=lambda x: x[1])
    
    for code, status in sorted_codes:
        click.echo(f"Code: {code.decode('utf-8')} - Status: {status.decode('utf-8')}")

cli.add_command(generateaccesscodes)
cli.add_command(listaccesscodes)

if __name__ == "__main__":
    cli()
