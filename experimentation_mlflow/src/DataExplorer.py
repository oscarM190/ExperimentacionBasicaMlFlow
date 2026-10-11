import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataExplorer:
    @staticmethod
    def explore_data(data):
        print(data.head().T)
        print(data.describe())
        print(data.info())
        missing_percent = data.isna().mean() * 100
        missing_percent.sort_values(ascending=False)
        print(f"Missing values:\n {missing_percent}")
        print(data.duplicated().sum())

    @staticmethod
    def plot_histograms(data):
        morfo_mapping = {
            'mean smoothness': 'smoothness_mean',
            'mean compactness': 'compactness_mean',
            'mean concavity': 'concavity_mean',
            'mean concave points': 'concave points_mean',
            'mean symmetry': 'symmetry_mean',
            'mean fractal dimension': 'fractal_dimension_mean'
        }
        morfo_vars = list(morfo_mapping.values())
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
        axes = axes.flatten()

        for i, var in enumerate(morfo_vars):
            sns.histplot(
                data=data,
                x=var,
                hue='diagnosis',
                multiple="stack",
                palette={'M': '#e74c3c', 'B': '#2ecc71'},
                ax=axes[i],
                kde=True # Añadimos la curva para ver mejor la tendencia
            )
            axes[i].set_title(f'Distribución de {var}', fontsize=12)
            axes[i].set_xlabel('')
        plt.tight_layout()
        plt.suptitle('Histogramas Apilados: Variables Morfológicas por Diagnóstico', y=1.02, fontsize=16)
        plt.show()

    @staticmethod
    def descriptive_statistics(data):
        num_df = data.select_dtypes(include=['float64'])
        print("Estadisticas descriptivas para variables numericas: \n")
        print(num_df.select_dtypes(include=['number']).agg([
            'count', 'mean', 'std', 'min',
            lambda x: x.quantile(0.25),
            'median',
            lambda x: x.quantile(0.75), 'max',
            'skew', 'kurt'
        ]))
        cat_df = data.select_dtypes(include=['object'])
        print(f"Estadisticas descriptivas para variables categoricas: \n {cat_df.describe()}")

    @staticmethod
    def plot_correlation_matrix(data):
        Tcorrelation = data.corr(method='pearson',numeric_only=True)
        plt.rcParams['figure.figsize'] = (20, 15)
        sns.heatmap(round(Tcorrelation,2), annot=True)
        mask = np.triu(np.ones(Tcorrelation.shape), k=1).astype(bool)
        df_pairs = Tcorrelation.where(mask).stack().reset_index()
        df_pairs.columns = ['Variable 1', 'Variable 2', 'Correlacion']
        high_corr = df_pairs[df_pairs['Correlacion'] > 0.9].sort_values(by='Correlacion', ascending=False)
        print(f"Correlacion entre variables: \n {high_corr}")

    def distibution_diagnosis_sex(data):
        dist_pct = data['diagnosis'].value_counts(normalize=True) * 100
        plt.figure(figsize=(8, 6))
        bars = plt.bar(dist_pct.index, dist_pct.values, color=['#3498db', '#e74c3c'])
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.title('Distribución Porcentual de Diagnósticos (M vs B)', fontsize=14, pad=20)
        plt.xlabel('Tipo de Diagnóstico (B = Benigno, M = Maligno)', fontsize=12)
        plt.ylabel('Porcentaje (%)', fontsize=12)
        plt.ylim(0, max(dist_pct.values) + 10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()