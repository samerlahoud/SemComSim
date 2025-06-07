import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Extracted data from actual experimental results (35 rounds)
rounds = list(range(1, 36))

# PROPORTIONAL FAIRNESS Results (from dump file)
prop_psnr = [18.05, 19.16, 19.75, 20.25, 20.75, 21.19, 21.56, 21.93, 22.35, 22.67,
             23.01, 23.40, 23.73, 24.07, 24.47, 24.84, 25.25, 25.62, 25.96, 26.29,
             26.57, 26.87, 27.16, 27.42, 27.69, 27.96, 28.15, 28.37, 28.60, 28.85,
             28.98, 29.15, 29.36, 29.43, 29.57]

prop_mse = [0.0157, 0.0121, 0.0106, 0.0094, 0.0084, 0.0076, 0.0070, 0.0064, 0.0058, 0.0054,
            0.0050, 0.0046, 0.0042, 0.0039, 0.0036, 0.0033, 0.0030, 0.0027, 0.0025, 0.0023,
            0.0022, 0.0021, 0.0019, 0.0018, 0.0017, 0.0016, 0.0015, 0.0015, 0.0014, 0.0013,
            0.0013, 0.0012, 0.0012, 0.0011, 0.0011]

prop_participation_gini = [0.400, 0.333, 0.222, 0.233, 0.200, 0.233, 0.195, 0.171, 0.170, 0.157,
                          0.148, 0.150, 0.144, 0.129, 0.124, 0.119, 0.125, 0.113, 0.109, 0.115,
                          0.102, 0.091, 0.093, 0.094, 0.089, 0.085, 0.090, 0.101, 0.092, 0.076,
                          0.081, 0.082, 0.077, 0.075, 0.078]

prop_effort_gini = [0.445, 0.342, 0.210, 0.219, 0.207, 0.239, 0.197, 0.166, 0.176, 0.157,
                   0.151, 0.149, 0.140, 0.123, 0.115, 0.124, 0.133, 0.126, 0.120, 0.127,
                   0.109, 0.095, 0.100, 0.102, 0.098, 0.093, 0.094, 0.101, 0.091, 0.079,
                   0.084, 0.090, 0.084, 0.082, 0.082]

# BASELINE Results (All clients participate)
baseline_psnr = [20.12, 21.82, 23.05, 24.36, 25.62, 26.79, 27.72, 28.50, 29.16, 29.66,
                 30.00, 30.25, 30.40, 30.50, 30.62, 30.67, 30.77, 30.83, 30.88, 30.93,
                 30.99, 31.09, 31.11, 31.24, 31.25, 31.32, 31.33, 31.42, 31.48, 31.49,
                 31.53, 31.55, 31.62, 31.64, 31.71]

baseline_mse = [0.0097, 0.0066, 0.0050, 0.0037, 0.0027, 0.0021, 0.0017, 0.0014, 0.0012, 0.0011,
                0.0010, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008,
                0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007,
                0.0007, 0.0007, 0.0007, 0.0007, 0.0007]

baseline_participation_gini = [0.000] * 35  # Perfect equality
baseline_effort_gini = [0.066] * 35  # Constant due to dataset size differences

# UTILITARIAN Results
util_psnr = [20.21, 22.39, 24.44, 25.96, 27.05, 28.18, 29.14, 29.60, 29.86, 30.25,
             30.43, 30.51, 30.60, 30.68, 30.74, 30.90, 30.98, 31.04, 31.12, 31.16,
             31.20, 31.27, 31.29, 31.30, 31.35, 31.45, 31.50, 31.52, 31.57, 31.54,
             31.65, 31.62, 31.68, 31.69, 31.75]

util_mse = [0.0095, 0.0058, 0.0036, 0.0025, 0.0020, 0.0015, 0.0012, 0.0011, 0.0010, 0.0009,
            0.0009, 0.0009, 0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008,
            0.0008, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007,
            0.0007, 0.0007, 0.0007, 0.0007, 0.0007]

util_participation_gini = [0.400, 0.250, 0.222, 0.208, 0.220, 0.211, 0.229, 0.179, 0.163, 0.167,
                          0.173, 0.167, 0.174, 0.190, 0.176, 0.167, 0.157, 0.144, 0.149, 0.137,
                          0.135, 0.118, 0.123, 0.126, 0.135, 0.136, 0.132, 0.129, 0.128, 0.118,
                          0.123, 0.112, 0.116, 0.116, 0.121]

util_effort_gini = [0.486, 0.332, 0.263, 0.244, 0.247, 0.242, 0.261, 0.213, 0.199, 0.207,
                   0.218, 0.211, 0.215, 0.231, 0.215, 0.204, 0.195, 0.182, 0.190, 0.185,
                   0.176, 0.160, 0.167, 0.169, 0.175, 0.176, 0.175, 0.169, 0.170, 0.160,
                   0.163, 0.155, 0.157, 0.155, 0.161]

# Final distributions (from dump file)
prop_final_participation = [15, 18, 26, 21, 22, 21, 19, 24, 21, 23]
prop_final_effort = [43530, 44136, 61256, 46641, 63514, 47334, 54872, 56496, 42945, 60674]

util_final_participation = [28, 19, 22, 15, 28, 19, 25, 21, 15, 18]
util_final_effort = [266984, 166736, 193192, 122155, 268491, 146510, 236816, 178904, 110430, 163556]

baseline_final_participation = [35, 35, 35, 35, 35, 35, 35, 35, 35, 35]
baseline_final_effort = [304710, 257460, 247380, 233205, 303135, 236670, 303240, 247170, 214725, 276990]

# Colors for consistency
colors = {
    'proportional': '#2E8B57',  # Sea green
    'utilitarian': '#DC143C',   # Crimson
    'baseline': '#4169E1'       # Royal blue
}

base_folder = './fed_semcom_v4_iot_fair_figures/'
# Figure 1: Performance Convergence
def create_performance_convergence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PSNR convergence
    ax1.plot(rounds, prop_psnr, 'o-', linewidth=2.5, markersize=4, 
             label='Proportional Fairness', color=colors['proportional'])
    ax1.plot(rounds, util_psnr, 's-', linewidth=2.5, markersize=4, 
             label='Utilitarian', color=colors['utilitarian'])
    ax1.plot(rounds, baseline_psnr, '^-', linewidth=2.5, markersize=4, 
             label='Baseline (All Clients)', color=colors['baseline'])
    
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Performance Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add final performance annotations
    ax1.annotate(f'{prop_psnr[-1]:.1f} dB', xy=(35, prop_psnr[-1]), 
                xytext=(32, prop_psnr[-1]-0.8),
                arrowprops=dict(arrowstyle='->', color=colors['proportional']), 
                fontsize=9, color=colors['proportional'])
    ax1.annotate(f'{util_psnr[-1]:.1f} dB', xy=(35, util_psnr[-1]), 
                xytext=(32, util_psnr[-1]+0.5),
                arrowprops=dict(arrowstyle='->', color=colors['utilitarian']), 
                fontsize=9, color=colors['utilitarian'])
    ax1.annotate(f'{baseline_psnr[-1]:.1f} dB', xy=(35, baseline_psnr[-1]), 
                xytext=(32, baseline_psnr[-1]-0.3),
                arrowprops=dict(arrowstyle='->', color=colors['baseline']), 
                fontsize=9, color=colors['baseline'])
    
    # MSE evolution (log scale)
    ax2.semilogy(rounds, prop_mse, 'o-', linewidth=2.5, markersize=4, 
                 label='Proportional Fairness', color=colors['proportional'])
    ax2.semilogy(rounds, util_mse, 's-', linewidth=2.5, markersize=4, 
                 label='Utilitarian', color=colors['utilitarian'])
    ax2.semilogy(rounds, baseline_mse, '^-', linewidth=2.5, markersize=4, 
                 label='Baseline (All Clients)', color=colors['baseline'])
    
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('MSE (Log Scale)')
    ax2.set_title('Mean Squared Error Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(base_folder + 'figure1_performance_convergence.png', dpi=300, bbox_inches='tight')
    plt.savefig(base_folder + 'figure1_performance_convergence.pdf', bbox_inches='tight')

# Figure 2: Fairness Evolution
def create_fairness_evolution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Participation Gini evolution
    ax1.plot(rounds, prop_participation_gini, 'o-', linewidth=2.5, markersize=4, 
             label='Proportional Fairness', color=colors['proportional'])
    ax1.plot(rounds, util_participation_gini, 's-', linewidth=2.5, markersize=4, 
             label='Utilitarian', color=colors['utilitarian'])
    ax1.plot(rounds, baseline_participation_gini, '^-', linewidth=2.5, markersize=4, 
             label='Baseline (All Clients)', color=colors['baseline'])
    
    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Participation Gini Coefficient')
    ax1.set_title('Participation Fairness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.text(30, 0.02, 'Perfect Equality', fontsize=9, alpha=0.7)
    
    # Effort Gini evolution
    ax2.plot(rounds, prop_effort_gini, 'o-', linewidth=2.5, markersize=4, 
             label='Proportional Fairness', color=colors['proportional'])
    ax2.plot(rounds, util_effort_gini, 's-', linewidth=2.5, markersize=4, 
             label='Utilitarian', color=colors['utilitarian'])
    ax2.plot(rounds, baseline_effort_gini, '^-', linewidth=2.5, markersize=4, 
             label='Baseline (All Clients)', color=colors['baseline'])
    
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Effort Gini Coefficient')
    ax2.set_title('Effort Fairness Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(base_folder + 'figure2_fairness_evolution.png', dpi=300, bbox_inches='tight')
    plt.savefig(base_folder + 'figure2_fairness_evolution.pdf', bbox_inches='tight')

# Figure 3: Final Distributions
def create_final_distributions():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    clients = list(range(1, 11))
    x = np.arange(len(clients))
    width = 0.25
    
    # Final participation distribution
    bars1 = ax1.bar(x - width, prop_final_participation, width, 
                    label='Proportional Fairness', color=colors['proportional'], alpha=0.8)
    bars2 = ax1.bar(x, util_final_participation, width, 
                    label='Utilitarian', color=colors['utilitarian'], alpha=0.8)
    bars3 = ax1.bar(x + width, baseline_final_participation, width, 
                    label='Baseline (All Clients)', color=colors['baseline'], alpha=0.8)
    
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Participation Count')
    ax1.set_title('Final Participation Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(clients)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Final effort distribution (in thousands)
    prop_effort_k = [x/1000 for x in prop_final_effort]
    util_effort_k = [x/1000 for x in util_final_effort]
    baseline_effort_k = [x/1000 for x in baseline_final_effort]
    
    bars1 = ax2.bar(x - width, prop_effort_k, width, 
                    label='Proportional Fairness', color=colors['proportional'], alpha=0.8)
    bars2 = ax2.bar(x, util_effort_k, width, 
                    label='Utilitarian', color=colors['utilitarian'], alpha=0.8)
    bars3 = ax2.bar(x + width, baseline_effort_k, width, 
                    label='Baseline (All Clients)', color=colors['baseline'], alpha=0.8)
    
    ax2.set_xlabel('Client ID')
    ax2.set_ylabel('Total Effort (×1000 training steps)')
    ax2.set_title('Final Effort Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(clients)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(base_folder + 'figure3_final_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(base_folder + 'figure3_final_distributions.pdf', bbox_inches='tight')

# Figure 4: Efficiency and Trade-off Analysis
def create_efficiency_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate total efforts and efficiencies
    prop_total_effort = sum(prop_final_effort)
    util_total_effort = sum(util_final_effort)
    baseline_total_effort = sum(baseline_final_effort)
    
    prop_efficiency = prop_psnr[-1] / (prop_total_effort / 1000)
    util_efficiency = util_psnr[-1] / (util_total_effort / 1000)
    baseline_efficiency = baseline_psnr[-1] / (baseline_total_effort / 1000)
    
    # Computational efficiency
    methods = ['Proportional\nFairness', 'Utilitarian', 'Baseline\n(All Clients)']
    efficiencies = [prop_efficiency, util_efficiency, baseline_efficiency]
    method_colors = [colors['proportional'], colors['utilitarian'], colors['baseline']]
    
    bars = ax1.bar(methods, efficiencies, color=method_colors, alpha=0.8, width=0.6)
    ax1.set_ylabel('PSNR per 1000 Training Steps')
    ax1.set_title('Computational Efficiency')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{eff:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Performance vs Fairness Trade-off
    ax2.scatter(prop_effort_gini[-1], prop_psnr[-1], s=200, color=colors['proportional'], 
               label='Proportional Fairness', marker='o', edgecolor='black', linewidth=2)
    ax2.scatter(util_effort_gini[-1], util_psnr[-1], s=200, color=colors['utilitarian'], 
               label='Utilitarian', marker='s', edgecolor='black', linewidth=2)
    ax2.scatter(baseline_effort_gini[-1], baseline_psnr[-1], s=200, color=colors['baseline'], 
               label='Baseline (All Clients)', marker='^', edgecolor='black', linewidth=2)
    
    ax2.set_xlabel('Effort Gini Coefficient (Lower = More Fair)')
    ax2.set_ylabel('Final PSNR (dB)')
    ax2.set_title('Performance vs Fairness Trade-off')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(base_folder + 'figure4_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(base_folder + 'figure4_efficiency_analysis.pdf', bbox_inches='tight')

# Figure 5: Summary Statistics Table
def create_summary_table():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Calculate summary statistics
    prop_total_effort = sum(prop_final_effort)
    util_total_effort = sum(util_final_effort)
    baseline_total_effort = sum(baseline_final_effort)
    
    prop_efficiency = prop_psnr[-1] / (prop_total_effort / 1000)
    util_efficiency = util_psnr[-1] / (util_total_effort / 1000)
    baseline_efficiency = baseline_psnr[-1] / (baseline_total_effort / 1000)
    
    # Create summary data
    summary_data = [
        ['Metric', 'Proportional\nFairness', 'Utilitarian', 'Baseline\n(All Clients)', 'Best Method'],
        ['Final PSNR (dB)', f'{prop_psnr[-1]:.2f}', f'{util_psnr[-1]:.2f}', f'{baseline_psnr[-1]:.2f}', 'Utilitarian'],
        ['Final MSE', f'{prop_mse[-1]:.4f}', f'{util_mse[-1]:.4f}', f'{baseline_mse[-1]:.4f}', 'Utilitarian/Baseline'],
        ['Total Effort (K)', f'{prop_total_effort/1000:.0f}', f'{util_total_effort/1000:.0f}', f'{baseline_total_effort/1000:.0f}', 'Proportional'],
        ['Participation Gini', f'{prop_participation_gini[-1]:.3f}', f'{util_participation_gini[-1]:.3f}', f'{baseline_participation_gini[-1]:.3f}', 'Baseline'],
        ['Effort Gini', f'{prop_effort_gini[-1]:.3f}', f'{util_effort_gini[-1]:.3f}', f'{baseline_effort_gini[-1]:.3f}', 'Baseline'],
        ['Efficiency (PSNR/K steps)', f'{prop_efficiency:.3f}', f'{util_efficiency:.3f}', f'{baseline_efficiency:.3f}', 'Proportional']
    ]
    
    # Create table
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.18, 0.18, 0.18, 0.21])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#E6E6FA')
                cell.set_text_props(weight='bold')
            elif j == 1:  # Proportional column
                cell.set_facecolor('#F0FFF0')
            elif j == 2:  # Utilitarian column
                cell.set_facecolor('#FFF0F5')
            elif j == 3:  # Baseline column
                cell.set_facecolor('#F0F8FF')
            elif j == 4:  # Best method column
                cell.set_facecolor('#F5F5DC')
    
    plt.title('FedLoL IoT: Comprehensive Performance Summary', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(base_folder + 'figure5_summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(base_folder + 'figure5_summary_table.pdf', bbox_inches='tight')

# Main execution
if __name__ == "__main__":
    print("Creating Figure 1: Performance Convergence...")
    create_performance_convergence()
    
    print("Creating Figure 2: Fairness Evolution...")
    create_fairness_evolution()
    
    print("Creating Figure 3: Final Distributions...")
    create_final_distributions()
    
    print("Creating Figure 4: Efficiency Analysis...")
    create_efficiency_analysis()
    
    print("Creating Figure 5: Summary Table...")
    create_summary_table()
    
    # Print comprehensive insights
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY:")
    print("="*60)
    
    prop_total_effort = sum(prop_final_effort)
    util_total_effort = sum(util_final_effort)
    baseline_total_effort = sum(baseline_final_effort)
    
    prop_efficiency = prop_psnr[-1] / (prop_total_effort / 1000)
    util_efficiency = util_psnr[-1] / (util_total_effort / 1000)
    baseline_efficiency = baseline_psnr[-1] / (baseline_total_effort / 1000)
    
    print(f"🎯 PERFORMANCE RANKING:")
    print(f"   1st: Utilitarian     - {util_psnr[-1]:.2f} dB")
    print(f"   2nd: Baseline        - {baseline_psnr[-1]:.2f} dB")  
    print(f"   3rd: Proportional    - {prop_psnr[-1]:.2f} dB")
    print()
    print(f"⚡ EFFICIENCY RANKING (PSNR per 1000 steps):")
    print(f"   1st: Proportional    - {prop_efficiency:.3f}")
    print(f"   2nd: Baseline        - {baseline_efficiency:.3f}")
    print(f"   3rd: Utilitarian     - {util_efficiency:.3f}")
    print()
    print(f"💰 COMPUTATIONAL COST:")
    print(f"   Proportional: {prop_total_effort/1000:.0f}K steps (Most efficient)")
    print(f"   Utilitarian:  {util_total_effort/1000:.0f}K steps ({util_total_effort/prop_total_effort:.1f}× more)")
    print(f"   Baseline:     {baseline_total_effort/1000:.0f}K steps ({baseline_total_effort/prop_total_effort:.1f}× more)")
    print()
    print(f"⚖️ FAIRNESS RANKING (Lower Gini = More Fair):")
    print(f"   1st: Baseline        - Participation: {baseline_participation_gini[-1]:.3f}, Effort: {baseline_effort_gini[-1]:.3f}")
    print(f"   2nd: Proportional    - Participation: {prop_participation_gini[-1]:.3f}, Effort: {prop_effort_gini[-1]:.3f}")
    print(f"   3rd: Utilitarian     - Participation: {util_participation_gini[-1]:.3f}, Effort: {util_effort_gini[-1]:.3f}")
    print()
    print(f"🔍 KEY INSIGHTS:")
    print(f"   • Utilitarian achieves highest performance ({util_psnr[-1]:.2f} dB) but at significant computational cost")
    print(f"   • Baseline provides excellent balance: near-optimal performance with perfect participation fairness")
    print(f"   • Proportional is most efficient: {prop_total_effort/baseline_total_effort:.1f}× less computation than baseline")
    print(f"   • Performance gap: Utilitarian vs Baseline = {util_psnr[-1]-baseline_psnr[-1]:.2f} dB")
    print(f"   • For resource-constrained IoT: Proportional offers best efficiency-performance trade-off")
    
    print("\nAll figures saved as both PNG (high-res) and PDF (vector) formats!")