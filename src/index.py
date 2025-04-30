import argparse

from src.query_set_generation.query_generator import QueryGenerator
from src.query_set_generation.groundings_generator import GroundingsGenerator
from src.query_set_generation.answer_generator import AnswerGenerator
from src.query_set_generation.reasonings_generator import ReasoningsGenerator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic_index', type=int, default = 0, required = False)
    parser.add_argument('--model_index', type=int, default = 6, required = False)
    parser.add_argument('--filename', type = str, default = '10-K_NVDA_20240128', required = False)
    parser.add_argument('--no_of_qstns', type = int, default = 5, required = False)

    args = parser.parse_args()

    #query generation
    print('Generate queries\n')
    query_gen = QueryGenerator(filename = args.filename, model_index = args.model_index)
    if args.topic_index == -1:
        for ti in range(4):
            query_gen.generate_query(no_of_qstns = args.no_of_qstns, topic_index = ti)
    else:
        query_gen.generate_query(no_of_qstns = args.no_of_qstns, topic_index = args.topic_index)
    query_gen.destroy()

    print('Generate groundings\n')
    groundings_gen = GroundingsGenerator(filename = args.filename, model_index = args.model_index)
    if args.topic_index == -1:
        for ti in range(4):
            groundings_gen.generate_groundings(topic_index = ti)
    else:
        groundings_gen.generate_groundings(topic_index = args.topic_index)
    groundings_gen.destroy()

    print('Generate answers\n')
    ans_gen = AnswerGenerator(filename = args.filename, model_index = args.model_index)
    if args.topic_index == -1:
        for ti in range(4):
            ans_gen.generate_answer(topic_index = ti)
    else:
        ans_gen.generate_answer(topic_index = args.topic_index)
    ans_gen.destroy()

    print('Generate reasonings\n')
    reason_gen = ReasoningsGenerator(filename = args.filename, model_index = args.model_index)
    if args.topic_index == -1:
        for ti in range(4):
            reason_gen.generate_reasonings(topic_index = ti)
    else:
        reason_gen.generate_reasonings(topic_index = args.topic_index)
    reason_gen.destroy()


