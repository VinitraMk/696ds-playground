import argparse
from src.query_set_generation.query_generator import QueryGenerator
from src.query_set_generation.answer_generator import AnswerGenerator
from src.query_set_generation.groundings_generator import GroundingsGenerator
from src.query_set_generation.reasonings_generator import ReasoningsGenerator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--model_index', type = int, default = 6, required = False)
    parser.add_argument('--topic_index', type = int, default = 0, required = False)
    parser.add_argument('--no_of_qstns', type = int, default = 10, required = False)

    args = parser.parse_args()

    if args.topic_index == -1:
        
        print('\n\nGeneration questions...')
        query_gen = QueryGenerator(filename = args.filename, model_index = args.model_index)
        for ti in range(4):
            query_gen.generate_query(no_of_qstns = args.no_of_qstns, topic_index = ti)
        query_gen.destroy()
        print('\n\nGeneration answers...')
        answer_gen = AnswerGenerator(filename = args.filename, model_index = args.model_index)
        for ti in range(4):
            answer_gen.generate_answer(topic_index = ti)
        answer_gen.destroy()

        print('\n\nGeneration groundings...')
        groundings_gen = GroundingsGenerator(filename = args.filename, model_index = args.model_index)
        for ti in range(4):
            groundings_gen.generate_groundings(topic_index = ti)
        groundings_gen.destroy()

        print('\n\nGeneration reasonings...')
        reasonings_gen = ReasoningsGenerator(filename = args.filename, model_index = args.model_index)
        for ti in range(4):
            reasonings_gen.generate_reasonings(topic_index = ti)
        reasonings_gen.destroy()
    else:
        print('\n\nGeneration questions...')
        query_gen = QueryGenerator(filename = args.filename, model_index = args.model_index)
        query_gen.generate_query(no_of_qstns = args.no_of_qstns, topic_index = args.topic_index)
        query_gen.destroy()

        print('\n\nGeneration answers...')
        answer_gen = AnswerGenerator(filename = args.filename, model_index = args.model_index)
        answer_gen.generate_answer(topic_index = args.topic_index)
        answer_gen.destroy()

        print('\n\nGeneration groundings...')
        groundings_gen = GroundingsGenerator(filename = args.filename, model_index = args.model_index)
        groundings_gen.generate_groundings(topic_index = args.topic_index)
        groundings_gen.destroy()

        print('\n\nGeneration reasonings...')
        reasonings_gen = ReasoningsGenerator(filename = args.filename, model_index = args.model_index)
        reasonings_gen.generate_reasonings(topic_index = args.topic_index)
        reasonings_gen.destroy()
