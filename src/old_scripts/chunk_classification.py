import argparse
import json
import numpy as np

SEED_METADATA_TOPICS = [
    "Risk Factors and Challenges",
    "Financial Performance and Metrics",
    "Business Operations, Strategy, and Market Positioning",
    "Market Trends, Economic Environment, and Industry Dynamics"
]

RELEVANCE_THRESHOLD = 3.0


class ChunkScorer:

    def __init__(self, chunk_filename):
        self.chunk_filename = chunk_filename

    def __score_chunks_by_topic(self, chunks, topic_threshold = np.ones(len(SEED_METADATA_TOPICS)) * RELEVANCE_THRESHOLD):
        instruction_prompt = f"""
        ### Task:
        Given a chunk of text and a topic, determine if the text is relevant to the topic.
        Respond with a score on a scale from 1 to 5, rating the relevance of the text to the topic.
        Consider the following rubric to rate the relevance of the text:
        1 = Completely irrelevant i.e content having no information about the topic
        2 = Slightly relevant i.e content vaguely mentioning the topic
        3 = Moderately relevant i.e content having some information about the topic
        4 = Highly relevant i.e content having good amount of information about the topic
        5 = Perfectly relevant i.e content being primarily about the topic provided

        ### Input format:
        - Topic
        - Text

        ### Output format:
        - relevance: "<score in scale 1-5>"
        - reasoning: "<short and concise explanation of relevance score>"

        ### Input:
        """

        relevant_chunks = []
        chunk_scores = []
        for ci, chunk in enumerate(chunks):
            relevance_scores = np.ones(len(SEED_METADATA_TOPICS))
            for ti,topic in enumerate(SEED_METADATA_TOPICS):
                prompt = instruction_prompt + f"\n- Topic: {topic}" + f"\n- Text: {chunk}"
                response = self.__execute_LLM_task(prompt, max_new_tokens=50).strip()
                #print('response: ', response)
                rsi = response.index("relevance: ")
                rei = response.index("\n")
                rel_score_str = response[rsi+11:rei]
                rel_score_str = re.findall(r'\d+', rel_score_str)
                if len(rel_score_str) > 0:
                    rel_score_str = rel_score_str[0]
                    rel_score = int(rel_score_str)
                else:
                    rel_score = 1
                relevance_scores[ti] = rel_score
            chunk_scores.append({
                'chunk_index': ci,
                'topics': SEED_METADATA_TOPICS,
                'relevance': relevance_scores.tolist()
            })
            if np.all((relevance_scores >= topic_threshold)==True):
                relevant_chunks.append(chunk)
                
        return relevant_chunks, chunk_scores

    def score_chunks(self):

        chunk_fp = f'data/chunked_data/{self.chunk_filename}.json'

        with open(chunk_fp, 'r') as fp:
            self.all_chunks = json.load(fp)['chunks']
        _, chunk_scores = self.__score_chunks_by_topic(chunks=self.all_chunks)
        #print('Filtered chunks: ', len(self.chunks))
        scored_chunk_fp = f'data/chunked_data/scored_chunks/{self.chunk_filename}.json'
        with open(scored_chunk_fp, 'w') as fp:
            json.dump(chunk_scores, fp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_filename', default = '10-K_AMD_20231230_chunked', type = str, required = False)
    args = parser.parse_args()

    chunk_scorer = ChunkScorer(args.chunk_filename)



