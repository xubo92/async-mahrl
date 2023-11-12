import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
matplotlib.rcParams.update({'font.size': 18})


# wf, fully-dec [-284, -355, -282, -340, -321]; [-1192, -1368, -1275, -1127, -1158]; (2490-1183)/(3315-1587); (99121-98130)/(241190-239093)
# wf, fully-dec-ctde [-281, -143, -272, -195, -252, -257, -317]; [-1256, -1442, -888, -844, -886, -862]
# wf, fully-cen [-257, -238]; [-800, -673, -621]
# wf, partial-cen [-379, -312]; [-1303, -1432]
# wf, partial-dec [-230.3, -288], [-675.2, -881]; (80869-80441)/(222236-220468) 
def td_scatter():
    x_data = [(66092 + 63429 + 66876 + 67152 + 67777)/5, 
               (66911 + 66159 + 65220 + 62410) / 4,
                (63974 + 64315 + 62518 + 61731 + 61040) /5,
                (62249 + 61502 + 60927 + 64940 + 58009) / 5,
                (179483 + 175380 + 162939 + 207433 + 179809)/5]
    y_data = [(15 + 11.0 + 13.8 + 12.8 + 12.0)/5,
                (17.0 + 18.0 + 18.0 + 17.0 + 17.0)/5,
                (11.8 + 17.0 + 12.0 + 14.0 + 13.8)/5,
                (9.89 + 13.8 + 13.8 + 13.0 + 11.8) / 5,
                (23.0 + 21.0 + 21.0 + 18.0 + 17.0) /5 ]
    hue_data = ["fully-dec", "fully-cen", "partial-cen", "sync-cut", "sync-wait"]
    Df = {"x_data":x_data, "y_data":y_data, "hue_data":hue_data, "style":hue_data}
    ax = sns.scatterplot(x="x_data", y="y_data", hue="hue_data", style="style", data=Df)
    ax.set(xlabel="Total Training Samples Until Convergence", ylabel="Evaluated Peak Performance")
    xticklabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_xticks()/1000]
    ax.set_xticklabels(xticklabels)
  
    ax.get_legend().set_title(None)
    plt.show()

# [[[], [], []], [[], [], []]]
def task_barplot():
    matplotlib.rc('xtick', labelsize=22)
    d11 = {"hue": ["Water-Filling", "Water-Filling", "Water-Filling", "Water-Filling", "Water-Filling", "Water-Filling", "Water-Filling"], "Methods":["sync-wait", "sync-wait", "sync-cut",  "sync-cut", "async(ours)", "async(ours)", "async(ours)"], "Averaged Step Reward (final policy)": [-281, -455, -1084, -1414, -284, -322, -282]}
    d12 = {"hue": ["Water-Filling", "Water-Filling", "Water-Filling", "Water-Filling", "Water-Filling", "Water-Filling"], "Methods":["sync-wait", "sync-wait", "sync_cut",  "async(ours)", "async(ours)", "async(ours)"], "Training Steps (until certain performance level)": [121203, 112266, 0, 18695, 20802, 17613]}
    d21 = {"hue": ["Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery"],"Methods":["sync-wait", "sync-wait", "sync-wait", "sync-wait", "sync-cut",  "sync-cut", "sync-cut",  "sync-cut", "async(ours)", "async(ours)", "async(ours)"], "Averaged Step Reward (final policy)": [9.8, 4.9, 10.8, 0.8, 9.39, 13.0, 7.8, 15.8, 12.8, 14.0, 13.0]}
    d22 = {"hue": ["Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery", "Tool-Delivery"],"Methods":["sync-wait", "sync-wait", "sync-wait", "sync-wait", "sync-cut",  "sync-cut", "sync-cut",  "sync-cut", "async(ours)", "async(ours)", "async(ours)"], "Training Steps (until certain performance level)": [56564, 60650, 25049, 67165, 25062, 19354, 16899,15863, 6039, 7180, 4774]}

    fig, axes = plt.subplots(2, 2, figsize=(15, 5), sharex=True , sharey=False)
    # add a big axes, hide frame
    ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.grid(False)
    # plt.xlabel("common X")
    
    fig.text(0.97, 0.5, 'Training Steps (until certain performance level)', ha='center', va='center', rotation='vertical', fontsize=22)
    fig.text(0.05, 0.5, 'Averaged Step Reward (final policy)', ha='center', va='center', rotation='vertical', fontsize=22)
    
    sns.set_theme(style="white", palette="Blues")
    ax11 = sns.barplot(ax=axes[0, 0], x=d11["Methods"], y=d11["Averaged Step Reward (final policy)"])
    ax11.ticklabel_format(style='sci', axis='y', scilimits=(0,0))



    ax12 = sns.barplot(ax=axes[0, 1], x=d12["Methods"], y=d12["Training Steps (until certain performance level)"])
    ax12.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))
    ax12.yaxis.set_label_position("right")
    ax12.yaxis.tick_right()
   
    sns.set_theme(style="white", palette="Greens")
    ax21 = sns.barplot(ax=axes[1, 0], x=d21["Methods"], y=d21["Averaged Step Reward (final policy)"])


    ax22 = sns.barplot(ax=axes[1, 1], x=d22["Methods"], y=d22["Training Steps (until certain performance level)"])

    ax22.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))
    ax22.yaxis.set_label_position("right")
    ax22.yaxis.tick_right()  
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="#C0D5E5", lw=8),
                    Line2D([0], [0], color="#C0DEBB", lw=8)]

    ax.legend(custom_lines, ['Water-Filling', 'Tool-Delivery'], loc='upper center', ncol=2, fontsize=22, bbox_to_anchor=(0.5, 1.15))
     
    plt.show()

    # assert len(wf_path) != 0 and len(td_path) != 0

    # # align these data  # ["sync-wait", "sync-cut", "fully-dec (async)"]
    # x_data = []
    # y_data = []
    # hue_data = []

    # for tp, task_path in enumerate([wf_path, td_path]):
    #     for i, it in enumerate(task_path):
    #         assert len(it) != 0 
    #         for j, jt in enumerate(it):
    #             df = pd.read_csv(jt)
    #             if tp == 0:
    #                 x_data.append("Water Filling")
    #             elif tp == 1:
    #                 x_data.append("Tool Delivery")
    #             y_data.append(np.mean(df["Value"].iloc[-1]))
    #             if i == 0:
    #                 hue_data.append("sync-wait")
    #             elif i == 1:
    #                 hue_data.append("sync-cut")
    #             elif i == 2:
    #                 hue_data.append("fully-dec (async)")

    # d = {"x_data":x_data , "y_data":y_data, "hue_data":hue_data}
    # Df = pd.DataFrame(data=d)

    # subDfwf = Df[Df["x_data"]=="Water Filling"]
    # subDfwf_min, subDfwf_max = subDfwf["y_data"].min(), subDfwf["y_data"].max()
    # subDfwf["y_data"] = (subDfwf["y_data"] - subDfwf_min) / (subDfwf_max - subDfwf_min)

    # subDftd = Df[Df["x_data"]=="Tool Delivery"]
    # subDftd_min, subDftd_max = subDftd["y_data"].min(), subDftd["y_data"].max()
    # subDftd["y_data"] = (subDftd["y_data"] - subDftd_min) / (subDftd_max - subDftd_min)

    # newDf = subDfwf.append(subDftd, ignore_index=True)
    # print(Df)

    # # ax = sns.barplot(x="x_data", y="y_data", hue="hue_data", data=Df)
    # ax = sns.barplot(x="x_data", y="y_data", hue="hue_data", data=newDf)
    # plt.show()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


    


def draw_curves(paths, legends, x_name="Training Steps (primitive)", y_name="Averaged Step Reward"):
    """
    # input: 1. data path (absolute) [[], [], ...] 2. legends [...], 3. x-axis name 4 y-axis name
    """
    assert len(paths) != 0 
    assert len(paths) == len(legends)
    
    # align these data
    x_data = []
    y_data = []
    hue_data = []

    y_data_smoothed = []

    # maxk = 200 # td env (200) 10 for pointplot
    # smooth_factor = 0.85 # td env
    # maxk = 400 # wf env
    # smooth_factor = 0.65 # wf env
    maxk = 400 # ct env
    smooth_factor = 0.85 # ct env

    x_ranges = []
    for i, it in enumerate(paths):
        assert len(it) != 0 
        for j, jt in enumerate(it):
            df = pd.read_csv(jt)
            df_smoothed = df.ewm(alpha=(1-smooth_factor)).mean() # use 0.65 for td env

            prim_step_list = df["Step"]
            prelim_values = df["Value"][:200]
            for k in range(maxk):
                # x_data.append(str(int(k * 8000 / 1000)) + "K") # td pointplot
                # good_id = find_nearest(prim_step_list, k * 8000) # td pointplot
                x_data.append(k * 500)
                good_id = find_nearest(prim_step_list, k * 500)
                if k == 0:
                    good_id = np.argmin(prelim_values)
                y_data.append(df["Value"][good_id])
                y_data_smoothed.append(df_smoothed["Value"][good_id])
                hue_data.append(legends[i])

    d = {'hue_data':hue_data, 'x_data':x_data, "y_data":y_data, "y_data_smoothed": y_data_smoothed}
    Df = pd.DataFrame(data=d)
    print(Df)


    # sns.set_theme(style="white", palette="hls")
    sns.set_theme(style="white", palette="husl")
    # ax = sns.lineplot(xw="x_data", y="y_data", hue='hue_data', data=Df, ci="sd")
    ax = sns.lineplot(x="x_data", y="y_data_smoothed", hue='hue_data', data=Df, ci="sd")
    # ax = sns.pointplot(x="x_data", y="y_data_smoothed", hue='hue_data', data=Df, ci="sd", capsize=.2, markers=["o", "^", "s", "d", "P"])
    # ax.set_xbound(lower=0.0, upper=50000) # td env
    # ax.set_xbound(lower=0.0, upper=200000) # wf env
    ax.set_xbound(lower=0.0, upper=200000) # ct env
    ax.set(xlabel=x_name, ylabel=y_name)
    ax.get_legend().set_title(None)


    # ax.legend(fontsize=20, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.15))  # wf env
    # ax.legend(fontsize=22, ncol=2, loc="best") # td env
    ax.legend(fontsize=22, ncol=1, loc="best") # ct env

    ax.xaxis.label.set_fontsize(24)
    ax.yaxis.label.set_fontsize(24)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'K'))
    
    plt.show()

# if __name__ == "__main__":
    # follow the order: end2end,  sync-wait, sync-cut, fully-dec
#     wf_path = [
# ["D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\sync-wait\\run-run2_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\sync-wait\\run-run3_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards.csv"], 

# ["D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\sync-cut\\run-run2_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards (1).csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\sync-cut\\run-run3_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards.csv"], 

# ["D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run1_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run3_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run5_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run6_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run7_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run8_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv"]]
    
#     td_path = [["D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-wait\\run-run6_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv", 
#             "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-wait\\run-run7_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
#             "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-wait\\run-run8_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
#             "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-wait\\run-run9_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
#             "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-wait\\run-run10_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv"], 
    
#     ["D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run15_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
#             "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run16_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
#             "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run17_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
#             "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run18_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
#             "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run19_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
#             "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run20_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv"], 
    
#     ["D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec-ctde\\run-run8_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
#                 "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec-ctde\\run-run12_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
#                 "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec-ctde\\run-run9_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
#                 "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec-ctde\\run-run10_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
#                 "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec-ctde\\run-run11_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv"]]
    # draw barplot
    # task_barplot(wf_path, td_path)
    # task_barplot()




if __name__ == "__main__":
    # tooldelivery env
    # paths = [
#         ["D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec\\run-run14_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec\\run-run15_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec\\run-run16_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec\\run-run17_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec\\run-run18_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv"],
       
#         ["D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec-ctde\\run-run8_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
#                 "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec-ctde\\run-run12_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
#                 "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec-ctde\\run-run9_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
#                 "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec-ctde\\run-run10_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
#                 "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-dec-ctde\\run-run11_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv"],
            
            

        # ["D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-cen\\run-run5_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
        #         "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-cen\\run-run6_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
        #         "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-cen\\run-run8_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
        #         "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-cen\\run-run9_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
        #         "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\fully-cen\\run-run10_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv"],
        
        # ["D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\partial-cen\\run-run6_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
        #         "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\partial-cen\\run-run7_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
        #         "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\partial-cen\\run-run8_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
        #         "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\partial-cen\\run-run9_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
        #         "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\partial-cen\\run-run10_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv"]]            
#         ["D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\partial-dec\\run-run5_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\partial-dec\\run-run6_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\partial-dec\\run-run7_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\partial-dec\\run-run8_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\partial-dec\\run-run9_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\partial-dec\\run-run10_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv"]
#                 ]
            
            # ["D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run15_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
            # "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run16_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
            # "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run17_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
            # "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run18_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
            # "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run19_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
            # "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-cut\\run-run20_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv"],
            
            
            # ["D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-wait\\run-run6_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv", 
            # "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-wait\\run-run7_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
            # "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-wait\\run-run8_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
            # "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-wait\\run-run9_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
            # "D:\\xubo92\\hrl-mappo-server\\paper_results\\tooldelivery\\sync-wait\\run-run10_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv"]]
    

    # legends = ["fully-dec", "fully-dec (CTDE)", "fully-cen", "partial-cen", "partial-dec"] 
    # legends = ["fully-dec (CTDE)", "partially-dec"] 
    # legends = ["fully-dec", "fully-dec (CTDE)", "partially-cen"] 
    # legends = ["fully-cen", "partially-cen"]
    # legends = ["fully-cen", "partially-dec"]
    # draw_curves(paths, legends)


    # waterfill env
#     paths = [
#         [
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run1_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run3_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run5_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run6_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run7_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec\\run-run8_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv"],
            
#             [
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec-ctde\\run-run19_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec-ctde\\run-run20_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec-ctde\\run-run21_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec-ctde\\run-run22_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-dec-ctde\\run-run23_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv"],
            
#             [
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-cen\\run-fully-cen_run1_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-cen\\run-run1_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\fully-cen\\run-run2_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv"],
            
#             [
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\partial-cen\\run-run2_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\partial-cen\\run-run3_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards.csv"],
           
#            ["D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\partial-dec\\run-run1_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards.csv",
#            "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\partial-dec\\run-run2_logs_agent0_average_step_rewards_agent0_average_step_rewards-tag-agent0_average_step_rewards.csv"]

            

# ["D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\sync-cut\\run-run2_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards (1).csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\sync-cut\\run-run3_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards.csv"],
            
#             ["D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\sync-wait\\run-run2_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\sync-wait\\run-run3_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards.csv"],
            
#             ["D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\end2end\\run-run1_logs_agent1_average_step_rewards_agent1_average_step_rewards-tag-agent1_average_step_rewards (1).csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\end2end\\run-run28_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\waterfill\\end2end\\run-run29_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv"]
#         ]
#     # legends = ["fully-dec", "fully-dec-CTDE", "fully-cen", "partially-cen", "partially-dec"]
#     # # legends = ["fully-cen"]
#     legends = ["async (ours)", "sync-cut", "sync-wait", "end2end"]
#     draw_curves(paths, legends)


    # td env for degree of partially-dec
#     paths = [
#         ["D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\degree of partially-dec\\run-run2_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\degree of partially-dec\\run-run3_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv"], 

# ["D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\degree of partially-dec\\run-run4_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\degree of partially-dec\\run-run5_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\degree of partially-dec\\run-run6_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\degree of partially-dec\\run-run7_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\degree of partially-dec\\run-run8_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\degree of partially-dec\\run-run9_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv",
# "D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\degree of partially-dec\\run-run10_logs_agent2_average_step_rewards_agent2_average_step_rewards-tag-agent2_average_step_rewards.csv"]]
#     legends = ["Higher Degree", "Lower Degree"]
#     draw_curves(paths, legends)


    paths = [
        ["D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\continuous option\\fully-cen\\run-run2_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
"D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\continuous option\\fully-cen\\run-run3_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv"],
        
        
        ["D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\continuous option\\fully-cen\\run-run4_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
"D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\continuous option\\fully-cen\\run-run8_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
"D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\continuous option\\fully-cen\\run-run9_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
"D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\continuous option\\fully-cen\\run-run10_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv"],
        
        
        ["D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\continuous option\\fully-cen\\run-run6_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv",
"D:\\xubo92\\hrl-mappo-server\\paper_results\\discussion\\continuous option\\fully-cen\\run-run7_logs_average_step_rewards_average_step_rewards-tag-average_step_rewards.csv"]
    ]

    legends = [r'$\sigma=0$', r'$\sigma=0.5$', r'$\sigma=0.8$']
    draw_curves(paths, legends)