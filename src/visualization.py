import matplotlib.pyplot as plt
import seaborn as sns

def print_preview(data, columns, n=5, col_width=20):

    max_cols = len(columns)

    header = ""
    for col in columns[:max_cols]:
        header += str(col)[:col_width].ljust(col_width) + " | "
    print(header)
    print("-" * len(header))

    for i in range(min(n, len(data))):
        row = ""
        for val in data[i, :max_cols]:
            s = str(val)
            if len(s) > col_width:
                s = s[:col_width-3] + "..."  # rút gọn cho đẹp

            row += s.ljust(col_width) + " | "
        print(row)


def describe(x, columns):
    import numpy as np

    x = x.astype(float)

    count = np.sum(~np.isnan(x), axis=0)
    mean  = np.nanmean(x, axis=0)
    std   = np.nanstd(x, axis=0, ddof=1)
    min_  = np.nanmin(x, axis=0)
    p25   = np.nanpercentile(x, 25, axis=0)
    p50   = np.nanpercentile(x, 50, axis=0)
    p75   = np.nanpercentile(x, 75, axis=0)
    max_  = np.nanmax(x, axis=0)

    stats = np.vstack([count, mean, std, min_, p25, p50, p75, max_])
    row_names = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    value_width = 12
    max_col_name_len = max(len(col) for col in columns)
    col_width = max(value_width, max_col_name_len + 2)

    row_label_width = 12
    header = " " * row_label_width + "".join([f"{col:<{col_width}}" for col in columns])
    print(header)
    print("-" * len(header))

    for row_name, row_vals in zip(row_names, stats):
        row_text = f"{row_name:<{row_label_width}}" + "".join(
            [f"{v:<{col_width}.4f}" for v in row_vals]
        )
        print(row_text)

def residual_plots(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, ax=axes[0], color='steelblue')

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_title(f'{model_name}: Predicted vs Actual')
    axes[0].set_xlabel('Actual Price (Log-scaled)')
    axes[0].set_ylabel('Predicted Price (Log-scaled)')
    

    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3, ax=axes[1], color='crimson')
    axes[1].axhline(0, color='black', linestyle='--', lw=2)
    axes[1].set_title(f'{model_name}: Residual Plot')
    axes[1].set_xlabel('Predicted Price')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    
    plt.tight_layout()
    plt.show()