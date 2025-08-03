from time import time

class EntityGrouper:


if __name__ == "__main__":
    st = time()

    log_fp = f'logs/bu-entity-grouping-logs.txt'
    log_file = open(log_fp, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file

    #multiprocessing.set_start_method("spawn")  # Fixes CUDA issue with multiprocessing
    #torch.cuda.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_of_entities', type = int, default = 5, required = False)
    parser.add_argument('--no_of_companies', type = int, default = 3, required = False)

    args = parser.parse_args()

    ground_gen = EntityGrouper()
    ground_gen.set_filename(args.filename)
    ground_gen.generate_groundings(skip_entity_sampling = args.skip_entity_sampling, no_of_entities = args.no_of_entities)

    print(f'\n\n### TIME TAKEN: {(time() - st)/60:.2f} mins')
    sys.stdout = old_stdout
    log_file.close()
    ground_gen.destroy()