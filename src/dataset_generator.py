import argparse
from src.query_set_generation.query_generator import QueryGenerator
from src.query_set_generation.answer_generator import AnswerGenerator
from groundings_generator import GroundingsGenerator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--model_index', type = int, default = 6, required = False)
    parser.add_argument('--no_of_qstns', type = int, default = 5, required = False)

    args = parser.parse_args()

    print('\n\nGeneration questions...')
    query_gen = QueryGenerator(model_index = args.model_index)
    query_gen.set_filename(args.filename)
    query_gen.generate_query()
    query_gen.destroy()

    print('\n\nGeneration answers...')
    answer_gen = AnswerGenerator(model_index = args.model_index)
    answer_gen.set_filename(args.filename)
    answer_gen.generate_answer()
    answer_gen.destroy()

    print('\n\nGeneration groundings...')
    groundings_gen = GroundingsGenerator(model_index = args.model_index)
    groundings_gen.set_filename(args.filename)
    groundings_gen.generate_groundings()
    groundings_gen.destroy()
