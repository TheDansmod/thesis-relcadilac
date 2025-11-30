
def create_plots_434():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    # import scienceplots

    # plt.style.use(['science', 'ieee'])
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "lines.linewidth": 1,
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman"], # Match IEEEtran font
        "text.usetex": True,                        # Use LaTeX for text rendering
        "pgf.rcfonts": False,                       # Disables Matplotlib's internal font handling
        "font.size": 10,                            # IEEEtran main text size
        "axes.labelsize": 10,
        "legend.fontsize": 8,                       # Legends are typically smaller
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (7.16, 4.0),              # Width matched to IEEEtran \textwidth (double column)
                                                    # Use (3.5, 3.5) if targeting a single column width.
    })
    shd_samples_ancestral_cmaes = [12.666667, 16.000000, 14.000000, 10.000000, 22.000000]  # 5 points - rest 4 points
    shd_samples_ancestral_relcadilac = [25.5, 25.5, 9.0, 19.0]
    shd_samples_ancestral_dcd = [14.6, 8.8, 13.2, 17.6]
    shd_samples_bowfree_cmaes = [20.000000, 25.666667, 14.333333, 30.000000, 21.000000]  # 5 points - rest 4 points
    shd_samples_bowfree_relcadilac = [36.2, 31.4, 30.8, 43.4]
    shd_samples_bowfree_dcd = [24.4, 16.8, 17.2, 22.0]

    shd_nodes_ancestral_cmaes = [10.333333, 14.000000, 26.666667, 38.000000, 33.666667]
    shd_nodes_ancestral_relcadilac = [8.4, 32.2, 88.0, 161.0, 297.4]
    shd_nodes_ancestral_dcd = [8.6, 16.0, 33.6, 42.2, 47.4]
    shd_nodes_bowfree_cmaes = [10.000000, 14.333333, 44.333333, 49.666667, 32.000000]
    shd_nodes_bowfree_relcadilac = [9.4, 33.2, 74.8, 83.8, 177.0]
    shd_nodes_bowfree_dcd = [8.0, 14.8, 42.2, 35.6, 39.0]

    pag_f1_samples_ancestral_cmaes = [0.8974333333333333, 0.8277333333333333, 0.8395666666666667, 0.9394, 0.8336]  # 5 points - rest 4 points
    pag_f1_samples_ancestral_relcadilac = [0.7473429951690821, 0.75, 0.8095238095238095, 0.8131868131868132]
    pag_f1_samples_ancestral_dcd = [0.7994949494949495, 0.8771209565495889, 0.7860300618921309, 0.8415491977739056]
    pag_f1_samples_ancestral_gfci = [0.7180128205128204, 0.7040809509775027, 0.7781491018559984, 0.5747528203670474]
    pag_f1_samples_bowfree_cmaes = [0.7824333333333332, 0.8602, 0.7972333333333333, 0.7649666666666667, 0.8516]  # 5 points - rest 4 points
    pag_f1_samples_bowfree_dcd = [0.8124048860890966, 0.7748197273896689, 0.8631442818784592, 0.8028984414278533]
    pag_f1_samples_bowfree_gfci = [0.4865196914886355, 0.6426720209614947, 0.6934934630586804, 0.5831000018443719]
    pag_f1_samples_bowfree_relcadilac = [0.804099520843707, 0.7376323214529678, 0.7637572820671412, 0.8376593401231082]

    pag_f1_nodes_ancestral_cmaes = [0.9037666666666667, 0.8395666666666667, 0.7599333333333332, 0.7014, 0.8021666666666668]
    pag_f1_nodes_ancestral_dcd = [0.8619841269841271, 0.8308809043764356, 0.8084291226281811, 0.8257679778818974, 0.802746024795019]
    pag_f1_nodes_ancestral_gfci = [0.8397619047619047, 0.762414709473533, 0.5640456689004716, 0.44626730836824297, 0.5791472017681996]
    pag_f1_nodes_ancestral_relcadilac = [0.9279411764705883, 0.6916910232452966, 0.6073716810463179, 0.5286846622942043, 0.3664625129599791]
    pag_f1_nodes_bowfree_cmaes = [0.9041333333333333, 0.7972333333333333, 0.8446666666666666, 0.7901333333333334, 0.7754666666666666]
    pag_f1_nodes_bowfree_dcd = [0.9055555555555556, 0.8146757186190957, 0.8534426697607447, 0.703412828229868, 0.7205882352941176]
    pag_f1_nodes_bowfree_gfci = [0.7611111111111111, 0.6285546624348418, 0.48913293087951554, 0.5614647893196933, 0.591304347826087]
    pag_f1_nodes_bowfree_relcadilac = [0.8613691090471276, 0.7852821986442675, 0.8045412777607556, 0.6438037256859717, 0.3596059113300492]

    fig, ax = plt.subplots(nrows=2, ncols=2)
    x1 = [500, 1000, 2000, 3000, 4000]
    x0 = [500, 1000, 2000, 4000]
    x2 = [5, 10, 15, 20, 30]

    ax[0][0].plot(x1, shd_samples_ancestral_cmaes, '-b')
    ax[0][0].plot(x0, shd_samples_ancestral_dcd, '--b')
    ax[0][0].plot(x0, shd_samples_ancestral_relcadilac, ':b')
    ax[0][0].plot(x1, shd_samples_bowfree_cmaes, '-g')
    ax[0][0].plot(x0, shd_samples_bowfree_dcd, '--g')
    ax[0][0].plot(x0, shd_samples_bowfree_relcadilac, ':g')
    # ax[0][0].set_xlabel('Sample Size')
    ax[0][0].set_ylabel(r'SHD \((\downarrow)\)')
    ax[0][0].grid(which="both", axis="both", linestyle='--', alpha=0.5)
    ax[0][0].tick_params(axis='x', which='both', labelbottom=False)

    ax[0][1].semilogy(x2, shd_nodes_ancestral_cmaes, '-b')
    ax[0][1].semilogy(x2, shd_nodes_ancestral_dcd, '--b')
    ax[0][1].semilogy(x2, shd_nodes_ancestral_relcadilac, ':b')
    ax[0][1].semilogy(x2, shd_nodes_bowfree_cmaes, '-g')
    ax[0][1].semilogy(x2, shd_nodes_bowfree_dcd, '--g')
    ax[0][1].semilogy(x2, shd_nodes_bowfree_relcadilac, ':g')
    # ax[0][1].set_xlabel('Number of Nodes')
    # ax[0][1].set_ylabel('SHD')
    ax[0][1].grid(which="both", axis="both", linestyle='--', alpha=0.5)
    ax[0][1].tick_params(axis='x', which='both', labelbottom=False)

    ax[1][0].plot(x1, pag_f1_samples_ancestral_cmaes, '-b')
    ax[1][0].plot(x0, pag_f1_samples_ancestral_dcd, '--b')
    ax[1][0].plot(x0, pag_f1_samples_ancestral_relcadilac, ':b')
    ax[1][0].plot(x0, pag_f1_samples_ancestral_gfci, '-.b')
    ax[1][0].plot(x1, pag_f1_samples_bowfree_cmaes, '-g')
    ax[1][0].plot(x0, pag_f1_samples_bowfree_dcd, '--g')
    ax[1][0].plot(x0, pag_f1_samples_bowfree_relcadilac, ':g')
    ax[1][0].plot(x0, pag_f1_samples_bowfree_gfci, '-.g')
    ax[1][0].set_xlabel('Sample Size')
    ax[1][0].set_ylabel(r'PAG Skeleton \(F_{1} (\uparrow)\)')
    ax[1][0].grid(which="both", axis="both", linestyle='--', alpha=0.5)

    ax[1][1].plot(x2, pag_f1_nodes_ancestral_cmaes, '-b')
    ax[1][1].plot(x2, pag_f1_nodes_ancestral_dcd, '--b')
    ax[1][1].plot(x2, pag_f1_nodes_ancestral_relcadilac, ':b')
    ax[1][1].plot(x2, pag_f1_nodes_ancestral_gfci, '-.b')
    ax[1][1].plot(x2, pag_f1_nodes_bowfree_cmaes, '-g')
    ax[1][1].plot(x2, pag_f1_nodes_bowfree_dcd, '--g')
    ax[1][1].plot(x2, pag_f1_nodes_bowfree_relcadilac, ':g')
    ax[1][1].plot(x2, pag_f1_nodes_bowfree_gfci, '-.g')
    ax[1][1].set_xlabel('Number of Nodes')
    # ax[1][1].set_ylabel(r'PAG Skeleton \(F_{1}\)')
    ax[1][1].grid(which="both", axis="both", linestyle='--', alpha=0.5)

    # Legend A: Encodes the Variable mapped to COLOR
    # We create lines with the specific colors but generic style (solid is usually best for color keys)
    legend_color_handles = [
        Line2D([0], [0], color='blue', lw=2, label=r'Ancestral'),
        Line2D([0], [0], color='green', lw=2, label=r'Bow-Free')
    ]
    
    # Legend B: Encodes the Variable mapped to LINE STYLE
    # We create lines with black color (neutral) to emphasize the style
    legend_style_handles = [
        Line2D([0], [0], color='black', lw=1.5, linestyle='--', label=r'DCD'),
        Line2D([0], [0], color='black', lw=1.5, linestyle='-.', label=r'GFCI'),
        Line2D([0], [0], color='black', lw=1.5, linestyle=':', label=r'Relcadilac'),
        Line2D([0], [0], color='black', lw=1.5, linestyle='-', label=r'CMA-ES')
    ]
    
    # 5. Placing the Legends
    # Note: When adding multiple legends to a figure, Matplotlib typically overwrites the first.
    # We must explicitly add the first legend artist back to the layout.
    
    # Primary Legend (Colors) - Positioned Upper Right of the Figure bounding box
    leg1 = fig.legend(handles=legend_color_handles, 
                      title=r"ADMG Classes",
                      loc='center left', 
                      bbox_to_anchor=(1, 0.65), # Coordinates (x, y) relative to figure
                      frameon=False) # Removing frame for cleaner IEEE look
    
    # Secondary Legend (Styles) - Positioned below the first
    leg2 = fig.legend(handles=legend_style_handles, 
                      title=r"Algorithms",
                      loc='center left', 
                      bbox_to_anchor=(1, 0.35),
                      frameon=False)
    
    # CRITICAL: If using axis-level legends, you would need ax.add_artist(leg1). 
    # Since we are using figure-level legends (fig.legend), Matplotlib handles them concurrently 
    # provided we don't overlap them, though older versions might require add_artist logic.
    # The 'bbox_inches' in savefig handles the extra width.
    # ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=np.arange(1, 10)))
    # 
    # # 2. Format as Scalar (Rational Integers)
    # # Instead of 10^1, we display 10.
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    # 
    # # 3. Innovative/Detailed: Annotate exact bounds
    # # We add manual ticks for the global min and max to show the exact range.
    # # We retrieve current ticks, append min/max, and re-apply.
    # # Note: This is an "out-of-the-box" method to mix dynamic and static ticks.
    # yticks = list(ax.get_yticks()) + [global_min, global_max]
    # # Filter ticks to ensure they are within the viewable plot limits to avoid compression
    # ylim = ax.get_ylim()
    # visible_ticks = [y for y in yticks if ylim[0] <= y <= ylim[1]]
    # # We can strictly set these, but usually, letting the Locator handle the
    # # log structure is safer. Instead, let's just ensure the minor grid works:
    # ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=100))

    plt.tight_layout()
    # plt.legend()
    plt.savefig("diagrams/to_include_in_paper/pag_skeleton_f1_shd_nodes_samples.pdf", dpi=1200, bbox_inches="tight")
    # plt.show()
