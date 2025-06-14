import json
import matplotlib.pyplot as plt
import os

def get_multi_hop_distribution_plot(query_store, plots_folder, filename):
    less_than_5hop_qstns = 0
    more_than_eq_5hop_qstns = 0
    total_qstns = 0
    qstn_distribution_by_entity = {}
    qstn_distribution_by_hopcount = {}

    for ek in query_store["queries"].keys():
        qstn_distribution_by_entity[ek] = { 'less_than_5hops': 0, 'more_than_eq_5hops': 0}
        for query_obj in query_store["queries"][ek]:
            #print(query_obj)
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
        total_qstns += 1

    # pie chart
    plot_path = os.path.join(plots_folder, f'{filename}_qstn_dist_overall.png')
    plt.pie(x = [less_than_5hop_qstns, more_than_eq_5hop_qstns], labels = ['less than 5 hops', 'more than or equal to 5 hops'], colors = ['r', 'b'], autopct='%1.1f%%', textprops = {'color': 'white'})
    plt.legend(loc='upper left', bbox_to_anchor=(-0.3, 1), fontsize=8)
    plt.title('Question distribution over whole dataset')
    plt.savefig(plot_path)
    plt.clf()

    # bar chart by hop count
    plot_path = os.path.join(plots_folder, f'{filename}_qstn_dist_by_hops.png')
    x_keys = list(qstn_distribution_by_hopcount.keys())
    y_vals = list(qstn_distribution_by_hopcount.values())
    plt.bar(x_keys, y_vals, width = 0.3, color = 'b', label='question count by chunks used / hops')
    plt.xticks(x_keys, [f'{x}' for x in x_keys])
    plt.xlabel('Chunks used / hop count')
    plt.ylabel('No of questions')
    plt.legend()
    plt.savefig(plot_path)

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


    print('\n No of questions with less than 5 chunks hop: ', less_than_5hop_qstns)

if __name__ == "__main__":
    filename = '10-K_NVDA_20240128'
    model_folder = 'llama'
    expt_name = 'single-grounding-per-entity-2'
    #expt_name = 'single-grounding-per-entity'
    #generated_dataset_path = f'data/queries/{model_folder}/{filename}_generated_queries.json'
    generated_dataset_path = f'final_results/experiments/exp-{expt_name}/{filename}_generated_queries.json'
    #plots_folder = f'figures/data/queries/{model_folder}'
    plots_folder = f'final_results/experiments/exp-{expt_name}'

    with open(generated_dataset_path, "r") as fp:
        query_store = json.load(fp)

    get_multi_hop_distribution_plot(query_store, plots_folder, filename) 
