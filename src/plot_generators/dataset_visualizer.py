import json
import matplotlib.pyplot as plt
import os
import seaborn as sns

def get_multi_hop_distribution_plot(query_store, plots_folder, filename):
    less_than_5hop_qstns = 0
    more_than_eq_5hop_qstns = 0
    total_qstns = 0
    qstn_distribution_by_entity = {}
    qstn_distribution_by_hopcount = {}
    qstn_distribution_by_doccount = {}
    qstn_distribution_by_tokencount = {}
    qstn_distribution_by_subtype = {
        'temporal_analysis': 0,
        'summarization': 0,
        'numerical_analysis': 0,
        'entity_interaction_analysis': 0,
        'event_interaction_analysis': 0
    }
    hop_bucket_size = 5
    token_bucket_size = 500
    bucketed_qstn_dist_hopcount = {}
    bucketed_qstn_dist_tokencount = {}
    min_tokens = float('inf')
    max_tokens = 0
    total_q = 0

    # Muted palette
    bar_color   = "#4C72B0"   # desaturated blue-gray
    edge_color  = "#2f3e4a"   # darker edge
    tick_color  = "#4a4a4a"   # neutral gray
    grid_color  = "#d9d9d9"   # light gray

    pie_colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]  # blue, muted red, muted green, muted purple


    for ek in query_store["queries"].keys():
        qstn_distribution_by_entity[ek] = { 'less_than_5hops': 0, 'more_than_eq_5hops': 0}
        total_q += len(query_store["queries"][ek])
        for query_obj in query_store["queries"][ek]:

            for st in query_obj['actual_query_type']:
                    qstn_distribution_by_subtype[st] += 1
            
            #print(query_obj)

            # question count by chunks used
            cu_count = len(query_obj['chunks_used'])
            if  cu_count < 5:
                qstn_distribution_by_entity[ek]['less_than_5hops'] += 1
                less_than_5hop_qstns += 1
            else:
                qstn_distribution_by_entity[ek]['more_than_eq_5hops'] += 1
                more_than_eq_5hop_qstns += 1
            if cu_count in qstn_distribution_by_hopcount:
                qstn_distribution_by_hopcount[cu_count] += 1
            else:
                qstn_distribution_by_hopcount[cu_count] = 1

            # question count by docs used
            
            d_count = len(query_obj['docs_considered'])
            if d_count in qstn_distribution_by_doccount:
                qstn_distribution_by_doccount[d_count] += 1
            else:
                qstn_distribution_by_doccount[d_count] = 1

            # question count by citation tokens used
            c_count = query_obj['citations_token_count']
            min_tokens = min(min_tokens, c_count)
            max_tokens = max(max_tokens, c_count)
            if c_count in qstn_distribution_by_tokencount:
                qstn_distribution_by_tokencount[c_count] += 1
            else:
                qstn_distribution_by_tokencount[c_count] = 1

        total_qstns += 1

    # organize into buckets
    for k, v in qstn_distribution_by_hopcount.items():
        
        bs = (k // hop_bucket_size) * hop_bucket_size
        be = bs + hop_bucket_size - 1
        bl = f"{bs}-{be}"
        if bl in bucketed_qstn_dist_hopcount:
            bucketed_qstn_dist_hopcount[bl] += v
        else:
            bucketed_qstn_dist_hopcount[bl] = v

    for k, v in qstn_distribution_by_tokencount.items():
        bs = (k // token_bucket_size) * token_bucket_size
        be = bs + token_bucket_size - 1
        bl = f"{bs/1000:.1f}k-{be/1000:.1f}k"
        #print(k, bl)
        if bl in bucketed_qstn_dist_tokencount:
            bucketed_qstn_dist_tokencount[bl] += v
        else:
            bucketed_qstn_dist_tokencount[bl] = v

    # pie chart
    plot_path = os.path.join(plots_folder, f'{filename}_qstn_dist_overall.png')
    plt.pie(x = [less_than_5hop_qstns, more_than_eq_5hop_qstns], labels = ['less than 5 hops', 'more than or equal to 5 hops'], colors = [pie_colors[1], pie_colors[0]], autopct='%1.1f%%', textprops = {'color': 'white'})
    plt.legend(loc='upper left', bbox_to_anchor=(0.8, 1.02), fontsize=7)
    plt.title('Question distribution over whole dataset')
    plt.savefig(plot_path)
    plt.clf()

    # bar chart by hop count
    sorted_buckets = sorted(bucketed_qstn_dist_hopcount.items(), key=lambda x: int(x[0].split('-')[0]))
    plot_path = os.path.join(plots_folder, f'{filename}_qstn_dist_by_hops.png')
    x_keys = [b[0] for b in sorted_buckets]
    y_vals = [b[1] for b in sorted_buckets]
    #x_keys = sorted(list(qstn_distribution_by_hopcount.keys()))
    #y_vals = [qstn_distribution_by_hopcount[k] for k in x_keys]
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_axisbelow(True)  # put grid in background
    plt.grid(axis='both', linestyle='--', linewidth=0.6, color=grid_color, alpha=0.8)
    plt.bar(x_keys, y_vals, width = 0.3, color = bar_color, edgecolor = edge_color, label='question count grouped by chunks used / hops')
    plt.xticks(x_keys, [f'{x}' for x in x_keys], rotation = 45, color = tick_color)
    plt.xlabel('Chunks used / hop count', color = tick_color)
    plt.ylabel('No of questions', color = tick_color)
    plt.title('Questions distributed by hop count')
    plt.tight_layout()
    plt.legend(loc = 'upper right', bbox_to_anchor=(1, 0.98), fontsize = 7)
    plt.savefig(plot_path)
    plt.clf()


    plot_path = os.path.join(plots_folder, f'{filename}_qstn_dist_by_doc_count.png')
    sorted_buckets = sorted(qstn_distribution_by_doccount.items(), key=lambda x: x[0])
    x_keys = [b[0] for b in sorted_buckets]
    y_vals = [b[1] for b in sorted_buckets]
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_axisbelow(True)  # put grid in background
    plt.grid(axis='both', linestyle='--', linewidth=0.6, color=grid_color, alpha=0.8)
    print(x_keys, y_vals)
    #sns.barplot(x=x_keys, y=y_vals, label='question count vs doc count')
    plt.bar(x_keys, y_vals, width = 0.3, color = bar_color, edgecolor = edge_color, label='question count grouped by doc count')
    plt.xticks(x_keys, [f'{x}' for x in x_keys])
    plt.xlabel('Count of docs used', color = tick_color)
    plt.ylabel('No of questions', color = tick_color)
    plt.title('Questions distributed by document count')
    plt.legend(loc = 'upper right', bbox_to_anchor=(1, 0.98), fontsize = 7)
    plt.savefig(plot_path)
    plt.clf()

    sorted_buckets = sorted(bucketed_qstn_dist_tokencount.items(), key=lambda x: float(x[0].split('-')[0][:-1])*1000)
    plot_path = os.path.join(plots_folder, f'{filename}_qstn_dist_by_token_count.png')
    x_keys = [b[0] for b in sorted_buckets]
    y_vals = [b[1] for b in sorted_buckets]
    #x_keys = sorted(list(qstn_distribution_by_hopcount.keys()))
    #y_vals = [qstn_distribution_by_hopcount[k] for k in x_keys]
    plt.figure(figsize=(10, 13))
    ax = plt.gca()
    ax.set_axisbelow(True)  # put grid in background
    plt.grid(axis='both', linestyle='--', linewidth=0.6, color=grid_color, alpha=0.8)
    #sns.barplot(x=x_keys, y=y_vals)
    plt.bar(x_keys, y_vals, width = 0.3, color = bar_color, edgecolor = edge_color, label='question count grouped by citations token count')
    plt.xticks(x_keys, [f'{x}' for x in x_keys], rotation = 45)
    plt.xlabel('Count of tokens in citations', color=tick_color)
    plt.ylabel('No of questions', color = tick_color)
    plt.title('Questions distributed by citations token count')
    plt.legend(loc = 'upper right', bbox_to_anchor=(1, 0.98), fontsize = 7)
    plt.savefig(plot_path)
    plt.clf()

    sorted_buckets = sorted(qstn_distribution_by_subtype.items(), key=lambda x: x[0])
    plot_path = os.path.join(plots_folder, f'{filename}_qstn_dist_by_subtype.png')
    x_keys = [b[0] for b in sorted_buckets]
    x_nums = list(range(len(qstn_distribution_by_subtype.keys())))
    y_vals = [b[1] for b in sorted_buckets]
    #x_keys = sorted(list(qstn_distribution_by_hopcount.keys()))
    #y_vals = [qstn_distribution_by_hopcount[k] for k in x_keys]
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_axisbelow(True)  # put grid in background
    plt.grid(axis='both', linestyle='--', linewidth=0.6, color=grid_color, alpha=0.8)
    #sns.barplot(x=x_keys, y=y_vals)
    plt.bar(x_nums, y_vals, width = 0.3, color = bar_color, edgecolor = edge_color, label='question count grouped by subtypes')
    plt.xticks(x_nums, x_keys, rotation = 20)
    plt.xlabel('Question sub-types', color=tick_color)
    plt.ylabel('No of questions', color = tick_color)
    plt.title('Questions distributed by sub-types')
    plt.legend(loc = 'upper right', bbox_to_anchor=(1, 0.98), fontsize = 7)
    plt.savefig(plot_path)
    plt.clf()


    print('Total no of questions formed: ', total_q)
    print('Minimum count of tokens used for questions: ', min_tokens)
    print('Maximum count of tokens used for questions: ', max_tokens)

    '''
    # grouped bar chart by entities
    plot_path = os.path.join(plots_folder, f'{filename}_qstn_dist_by_entities.png')
    fig, axs = plt.subplots(2, 2)
    x_keys = list(query_store["queries"].keys())
    #print(list(qstn_distribution_by_entity.values()))
    y_val_1 = [tp['less_than_5hops'] for tp in list(qstn_distribution_by_entity.values())]
    y_val_2 = [tp['more_than_eq_5hops'] for tp in list(qstn_distribution_by_entity.values())]
    axs[0, 0].bar(x_keys[:5], y_val_1[:5], width=0.3, color='r', label='less than 5 hops')
    axs[0, 0].bar(x_keys[:5], y_val_2[:5], width=0.3, color='b', label='more than or equal to 5 hops')
    #axs[0, 0].set_xlabel('Entities')
    axs[0, 0].tick_params(axis='x', which='major', labelsize=6, labelrotation=45)
    axs[0, 0].tick_params(axis='y', which='major', labelsize=6)
    #axs[0, 0].set_ylabel('Distribution of <5 hops and >=5 hops questions')
    axs[0, 1].bar(x_keys[5:10], y_val_1[5:10], width=0.3, color='r')
    axs[0, 1].bar(x_keys[5:10], y_val_2[5:10], width=0.3, color='b')
    #axs[0, 1].set_xlabel('Entities')
    axs[0, 1].tick_params(axis='x', which='major', labelsize=6, labelrotation=45)
    axs[0, 1].tick_params(axis='y', which='major', labelsize=6)
    #axs[0, 1].set_ylabel('Distribution of <5 hops and >=5 hops questions')
    axs[1, 0].bar(x_keys[10:15], y_val_1[10:15], width=0.3, color='r')
    axs[1, 0].bar(x_keys[10:15], y_val_2[10:15], width=0.3, color='b')
    #axs[1, 0].set_xlabel('Entities')
    axs[1, 0].tick_params(axis='x', which='major', labelsize=6, labelrotation=45)
    axs[1, 0].tick_params(axis='y', which='major', labelsize=6)
    #axs[1, 0].set_ylabel('Distribution of <5 hops and >=5 hops questions')
    axs[1, 1].bar(x_keys[15:20], y_val_1[15:20], width=0.3, color='r')
    axs[1, 1].bar(x_keys[15:20], y_val_2[15:20], width=0.3, color='b')
    #axs[1, 1].set_xlabel('Entities')
    axs[1, 1].tick_params(axis='x', which='major', labelsize=6, labelrotation=45)
    axs[1, 1].tick_params(axis='y', which='major', labelsize=6)
    #axs[1, 1].set_ylabel('Distribution of <5 hops and >=5 hops questions')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_transform=fig.transFigure, fontsize=8)
    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(plot_path)
    '''


    print('\n No of questions with less than 5 chunks hop: ', less_than_5hop_qstns)

if __name__ == "__main__":

    # Global seaborn style
    #sns.set_theme(style="whitegrid")  # better background, clean axis
    #sns.set_palette("deep")  # cleaner colors
    #FONT_SCALE = 1.3
    #sns.set_context("paper", font_scale=FONT_SCALE)


    #filename = '10-K_NVDA_20240128'
    filecode = 'NVDA'
    model_folder = 'llama'
    dataset_root_fp = f'final_results/pipelines/multi-doc-unquantized-final'
    #expt_name = 'single-grounding-per-entity-5'
    #expt_name = 'single-grounding-per-entity'
    generated_dataset_path = f'{dataset_root_fp}/multi_doc_generated_queries.json'
    #generated_dataset_path = f'final_results/experiments/exp-{expt_name}/{filename}_generated_queries.json'
    #plots_folder = f'figures/data/queries/{model_folder}'
    #plots_folder = f'final_results/experiments/exp-{expt_name}'
    #plots_folder = f'final_results/pipelines/multi-doc-unquantized'

    with open(generated_dataset_path, "r") as fp:
        query_store = json.load(fp)

    get_multi_hop_distribution_plot(query_store, dataset_root_fp, 'multi_doc') 
