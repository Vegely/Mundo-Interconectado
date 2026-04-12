import networkx as nx
import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter
import numpy as np
from scipy.special import zeta as hurwitz_zeta, gammaincc, gamma, erfc, gammaln
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import poisson

def graphStatistics(G):
    nnodes = G.vcount()
    nedges = G.ecount()
    density = G.density()
    
    G_weak = G.connected_components(mode="weak")
    G_strong = G.connected_components(mode="strong")
    G_giant = G_strong.giant()
    
    degrees = G.degree(mode="in")
    avg_degree = sum(degrees) / G.vcount()
    
    diameter = G_giant.diameter(directed=True)
    rad_in = G_giant.radius(mode="in")
    rad_out = G_giant.radius(mode="out")
    
    print(f"Número de nodos (paquetes): {nnodes}")
    print(f"Número de aristas (dependencias): {nedges}")
    print(f"Densidad: {density:.6f}")
    
    print(f"Componentes débilmente conexas: {len(G_weak)}")
    print(f"Tamaño de la componente gigante: {max(G_weak.sizes())} paquetes ({(max(G_weak.sizes())/nnodes)*100:.1f}%)")
    print(f"Compnentes fuertemente conexas (bucles): {len(G_strong)}")
    
    print(f"Grado medio: {avg_degree:.2f}")
    print(f"Diámetro (C. gigante): {diameter}")
    print(f"Radio de entrada (C. gigante): {rad_in}")
    print(f"Radio de salida (C. gigante): {rad_out}")
    print("")
    
    return G_giant, G_weak, G_strong, nnodes, nedges, density, degrees, avg_degree, diameter, rad_in, rad_out

def isolatedComponents(G):
    G_weak = G.connected_components(mode="weak")
    giant_size = max(G_weak.sizes())    
    
    islands = {}
    for i, island in enumerate(G_weak):
        if len(island) != giant_size:
            isolated_packages = [G.vs[node]["id"] for node in island]
            islands[i] = isolated_packages
            
    print(f"Islas: {islands}")
    print("")
            
    return islands

def findCycle(G):
    G_strong = G.connected_components(mode="strong")
    
    cycles = [island for island in G_strong if len(island) > 1] # Los ciclos tienen que tener más de un paquete
    
    mean_length = sum(len(c) for c in cycles) / len(cycles)
    max_length = max(len(c) for c in cycles)
    
    cycle_packages = {}
    for i, cycle in enumerate(cycles):
        if len(cycle) > 0:
            packages = [G.vs[node]["id"] for node in cycle]
            cycle_packages[i] = packages
            
            with open(f"cycle_{i}_packages.txt", "w", encoding="utf-8") as f:
                for package in packages:
                    f.write(f"{package}\n")
                    
    print(f"Número de bucles: {len(G_strong)}")
    print(f"Bucles con más de un nodo: {len(cycles)}")
    print(f"Longitud media de bucle: {mean_length:.2f}")
    print(f"Longitud máxima de bucle: {max_length}")
    
    return packages

def freescaleTests(G):
    def alphaMLE(degrees, k_min):
        tail = degrees[degrees >= k_min]
        N = len(tail)
        
        if N < 50:
            return None, None
        
        alpha = 1 + N / np.sum(np.log(tail / (k_min - 0.5)))
        sigma = (alpha - 1) / math.sqrt(N)
        return alpha, sigma

    def empiricCDF(tail):
        """CDF empírica: P(X <= k)"""
        K   = np.sort(np.unique(tail))
        n    = len(tail)
        cdf  = np.array([np.sum(tail <= k) / n for k in K])
        return K, cdf
    
    def theoricCDF(K, k_min, alpha):
        """CDF teórica discreta: P(X <= k) = 1 - zeta(alpha,k+1)/zeta(alpha,k_min)"""
        zeta_kmin = hurwitz_zeta(alpha, k_min)
        return np.array([1 - hurwitz_zeta(alpha, k + 1) / zeta_kmin for k in K])

    def ksStat(degrees, k_min, alpha):
        """Estadístico KS entre CDF empírica y teórica discreta en la cola."""
        tail = degrees[degrees >= k_min]
        
        if len(tail) < 10:
            return np.inf
        
        K, cdf_emp = empiricCDF(tail)
        cdf_theor = theoricCDF(K, k_min, alpha)
        return np.max(np.abs(cdf_emp - cdf_theor))

    def kminOptSearch(degrees):
        """
        Prueba todos los valores únicos como k_min candidato.
        Sin límite de percentil — sigue el paper exactamente.
        """
        degrees_filtered = degrees[degrees > 0]
        degrees_unique = np.unique(degrees_filtered)
    
        ks_opt = np.inf
        kmin_opt = None
        alpha_opt = None
        sigma_opt = None
        res = []
    
        for k_min in degrees_unique:
            alpha, sigma = alphaMLE(degrees_filtered, k_min)
            
            # El paper dice n >= 50 en la cola (página 8)
            if alpha is None or alpha <= 1:
                continue
    
            ks_new = ksStat(degrees_filtered, k_min, alpha)
            res.append((k_min, alpha, sigma, ks_new))
    
            if ks_new < ks_opt:
                ks_opt    = ks_new
                kmin_opt  = k_min
                alpha_opt = alpha
                sigma_opt = sigma
    
        return kmin_opt, alpha_opt, sigma_opt, ks_opt, res
    
    def bootstrapKmin(degrees, n_boot=1000):
        """
        Bootstrap para estimar incertidumbre de xmin y alpha.
        Sección 3.5 del paper (Clauset et al. 2009).
        
        Para cada repetición:
            1. Muestrea n datos con reemplazamiento del dataset original
            2. Busca kmin óptimo en ese dataset
            3. Calcula alpha con ese kmin
        
        La desviación estándar de los 1000 estimados es la incertidumbre.
        """
        degrees_filtered = degrees[degrees > 0]
        n = len(degrees_filtered)
    
        kmin_boots  = []
    
        for _ in range(n_boot):
            boot = np.random.choice(degrees_filtered, size=n, replace=True)
            kmin_b, _, _, _, _ = kminOptSearch(boot)
    
            if kmin_b is None:
                continue
    
            kmin_boots.append(kmin_b)
    
        kmin_boots  = np.array(kmin_boots)
        
        return np.mean(kmin_boots), np.std(kmin_boots), kmin_boots,
        
    def generateTheoricTail(n, k_min, alpha):
        """
        Genera n muestras de power law discreta aproximada.
        Usa Eq. (D.6) del paper: x = floor((k_min - 0.5)*(1-r)^{-1/(alpha-1)} + 0.5)
        Filtra los que quedan por debajo de k_min por redondeo.
        """
        samples = []
        while len(samples) < n:
            nleft = n - len(samples)
            u = np.random.uniform(0, 1, int(nleft * 1.2) + 10)
            x = np.floor((k_min - 0.5) * (1 - u) ** (-1 / (alpha - 1)) + 0.5).astype(int)
            x = x[x >= k_min]
            samples.extend(x.tolist())
        return np.array(samples[:n])

    def getPValue(degrees, k_min, alpha, n_sim=2500):
        """
        P-value según Sección 4 del paper (Clauset et al. 2009).
        
        Procedimiento:
        1. Calcula KS empírico con los datos reales
        2. Genera n_sim datasets semiparamétricos:
           - Cola (k >= k_min): muestras de power law con alpha ajustado
           - Cuerpo (k < k_min): bootstrap de datos reales
        3. Reajusta alpha en cada dataset sintético
        4. p = fracción de KS sintéticos >= KS empírico
        
        Si p > 0.1: power law es hipótesis plausible
        Si p <= 0.1: power law se rechaza
        """
        n_total  = len(degrees)
        tail = degrees[degrees >= k_min]
        body = degrees[degrees < k_min]
        n_tail   = len(tail)
        n_body = n_total - n_tail
    
        ks_emp = ksStat(degrees, k_min, alpha)
    
        count = 0
        valid = 0
    
        for _ in range(n_sim):
            n_tail_sim   = np.random.binomial(n_total, n_tail / n_total)
            n_body_sim = n_total - n_tail_sim
    
            if n_tail_sim < 10:
                continue
            tail_sim = generateTheoricTail(n_tail_sim, k_min, alpha)
            body_sim = np.random.choice(body, n_body_sim, replace=True)
            sim = np.concatenate([tail_sim, body_sim]).astype(float)
    
            k_min_sim, alpha_sim, _, _, _ = kminOptSearch(sim)
            
            if k_min_sim is None or alpha_sim is None:
                continue
    
            ks_sim = ksStat(sim, k_min_sim, alpha_sim)
            if ks_sim >= ks_emp:
                count += 1
            valid += 1
    
        return count / valid if valid > 0 else np.nan
    
    in_deg  = np.array(G.degree(mode="in"),  dtype=np.float64)
    out_deg = np.array(G.degree(mode="out"), dtype=np.float64)

    results = {}
    
    for name, degrees in [("IN", in_deg), ("OUT", out_deg)]:
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")
    
        degrees_filtered = degrees[degrees > 0]
        
        kmin, alpha, sigma, ks, all_res = kminOptSearch(degrees_filtered)
        tail = degrees_filtered[degrees_filtered >= kmin]
        n_tail = len(tail)
        pct_tail = 100 * n_tail / len(degrees_filtered)
   
        print(f"k_min óptimo: {kmin}")
        print(f"Alpha óptimo: {alpha:.4f} ± {sigma:.4f}")
        print(f"KS mínimo {ks:.4f}")
        print(f"Nodos en la cola: {n_tail} ({pct_tail:.1f}%)")
        
        kmin_mean, kmin_std, kmin_boots = bootstrapKmin(degrees_filtered, n_boot=1000)
        print(f"k_min bootstrap: {kmin_mean:.1f} ± {kmin_std:.1f}")
    
        pval = getPValue(degrees_filtered, kmin, alpha, n_sim=2500)
    
        if np.isnan(pval):
            msg = "No calculable"
        elif pval > 0.1:
            msg = f"Ley de potencias no rechazada (p={pval:.3f} > 0.1)"
        else:
            v = f"Ley de potencias rechazada (p={pval:.3f} ≤ 0.1)"
    
        print(f"p-valor: {pval:.3f}")
        print(f"{msg}")
    
        results[name] = {
        "degrees"   : degrees_filtered,
        "kmin"      : kmin,
        "alpha"     : alpha,
        "sigma"     : sigma,
        "kmin_std"  : kmin_std,
        "pval"      : pval,
        "all_res"   : all_res
        }
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Ley de potencias",
                 fontsize=15, fontweight="bold")
    
    colors = {"IN": "royalblue", "OUT": "tomato"}
    
    for row, (name, data) in enumerate(results.items()):
        deg_v = data["degrees"]
        kmin = data["kmin"]
        alpha = data["alpha"]
        sigma = data["sigma"]
        kmin_std = data["kmin_std"]
        pval = data["pval"]
        all_res = data["all_res"]
        color = colors[name]
        
        tail_emp = deg_v[deg_v >= kmin]
        pct_tail = len(tail_emp) / len(deg_v)

        ax = axs[row, 0]
        counts = Counter(deg_v)
        k_vals = np.array(sorted(counts.keys()), dtype=float)
        pdf_emp = np.array([counts[k] for k in k_vals]) / len(deg_v)
        ax.loglog(k_vals, pdf_emp, 'o', color=color, alpha=0.5,
                  markersize=4, label="PDF empírica")
    
        k_fit = k_vals[k_vals >= kmin]
        zeta_kmin = hurwitz_zeta(alpha, kmin)
        pdf_fit = np.array([(hurwitz_zeta(alpha, k) - hurwitz_zeta(alpha, k + 1)) / zeta_kmin for k in k_fit]) * pct_tail
        ax.loglog(k_fit, pdf_fit, '-', color='black', lw=2,
                  label=f"α={alpha:.2f}±{sigma:.2f}\nk_min={kmin:.0f}±{kmin_std:.1f}")
    
        ax.axvline(x=kmin, color='gray', linestyle='--', alpha=0.7)
    
        pval_str = f"p={pval:.3f}" if not np.isnan(pval) else "p=N/A"
        msg = "No rechazada" if (not np.isnan(pval) and pval > 0.1) else "Rechazada"
        color_t = "seagreen" if msg == "No rechazada" else "crimson"
    
        ax.set_title(f"{name} - PDF\n{pval_str} → Ley de potencias {msg}",
                     fontsize=12, fontweight="bold", color=color_t)
        ax.set_xlabel("Grado k")
        ax.set_ylabel("P(k)")
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
    
        ax2    = axs[row, 1]
        t_arr  = np.array(all_res)
        ax2.plot(t_arr[:, 0], t_arr[:, 3], color=color, lw=1.5)
        ax2.axvline(x=kmin, color='black', linestyle='--', lw=2,
                    label=f"k_min óptimo={kmin:.0f}")
        ax2.scatter([kmin], [min(t_arr[:, 3])], color='black', s=80, zorder=5)
        ax2.set_xlabel("k")
        ax2.set_ylabel("Estadístico KS")
        ax2.set_title(f"{name} — Búsqueda de k_min", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

def powerLawTest(G):
    def llPowerlawDiscrete(x, k_min, alpha):
        zeta_kmin = hurwitz_zeta(alpha, k_min)
        return -alpha * np.log(x) - np.log(zeta_kmin)
    
    def llExponentialDiscrete(x, k_min, lam):
        return np.log(1 - np.exp(-lam)) + lam * k_min - lam * x
    
    def llYuleDiscrete(x, k_min, alpha):
        return np.log(alpha - 1) + gammaln(k_min + alpha - 1) - gammaln(k_min) + gammaln(x) - gammaln(x + alpha)
    
    def llPoissonDiscrete(x, k_min, mu):   
        return (x * np.log(mu) - gammaln(x + 1)) - np.log(np.exp(mu) - np.sum([mu**k / gamma(k + 1) for k in range(0, int(k_min))]))
    
    def fitExponentialDiscrete(tail, k_min):
        def negLl(lam):
            ll_array = llExponentialDiscrete(tail, k_min, lam)
            return -np.sum(ll_array)
        
        res = minimize_scalar(negLl, bounds=(1e-5, 10.0), method='bounded')
        return res.x if res.success else 1e-3
    
    def fitYuleDiscrete(tail, k_min):
        def negLl(alpha):
            ll_array = llYuleDiscrete(tail, k_min, alpha)
            return -np.sum(ll_array)
        
        res = minimize_scalar(negLl, bounds=(1.001, 10.0), method='bounded')
        return res.x if res.success else 2.0
    
    def fitPoissonDiscrete(tail, k_min):
        def negLl(mu):
            ll_array = llPoissonDiscrete(tail, k_min, mu)
            return -np.sum(ll_array)
        
        max_bound = max(100.0, np.mean(tail) * 3)
        res = minimize_scalar(negLl, bounds=(1e-5, max_bound), method='bounded')
        return res.x if res.success else np.mean(tail)
    
    def vuongTest(ll1, ll2):
        """
        ll1, ll2: arrays de log-likelihoods por observación para dos modelos.
        R > 0 → modelo 1 mejor; R < 0 → modelo 2 mejor.
        p-value: si p < 0.1 el signo es fiable.
        """
        n = len(ll1)
        diff = ll1 - ll2
        R = np.sum(diff)
        sigma2 = np.var(diff, ddof=1)    
        stat = R / np.sqrt(n * sigma2)
        pval = float(erfc(np.abs(stat) / np.sqrt(2)))
        return R, stat, pval
    
    def likelihoodRatioTests(degrees, k_min, alpha):
        tail = degrees[degrees >= k_min].astype(float)
        ll_pl = llPowerlawDiscrete(tail, k_min, alpha)
    
        results = {}
    
        lam_exp = fitExponentialDiscrete(tail, k_min)
        ll_exp = llExponentialDiscrete(tail, k_min, lam_exp)
        R_exp, stat_exp, p_exp = vuongTest(ll_pl, ll_exp)
        results["Exponential"] = (stat_exp, p_exp)
    
        mu_pois = fitPoissonDiscrete(tail, k_min)
        ll_pois = llPoissonDiscrete(tail, k_min, mu_pois)
        R_pois, stat_pois, p_pois = vuongTest(ll_pl, ll_pois)
        results["Poisson"] = (stat_pois, p_pois)
    
        alpha_yule = fitYuleDiscrete(tail, k_min)
        ll_yule = llYuleDiscrete(tail, k_min, alpha_yule)
        R_yule, stat_yule, p_yule = vuongTest(ll_pl, ll_yule)
        results["Yule"] = (stat_yule, p_yule)
    
        return results
    
    in_deg  = np.array(G.degree(mode="in"),  dtype=np.float64)
    out_deg = np.array(G.degree(mode="out"), dtype=np.float64)
    
    print(f"\n{'='*50}")
    print(f"Test de razón de verosimilitud")
    print(f"{'='*50}\n")
    
    header = f"{'Distribución':<18} {'LR':>10} {'p':>8}  {'Interpretación'}"
    sep = "-" * 65
    
    for deg_name, deg_arr, kmin_val, alpha_val, pval_mc in [
        ("IN",  in_deg,  results["IN"]["kmin"],  results["IN"]["alpha"],  results["IN"]["pval"]),
        ("OUT", out_deg, results["OUT"]["kmin"], results["OUT"]["alpha"], results["OUT"]["pval"]),
    ]:
        print(f"{deg_name}  |  k_min={kmin_val:.0f}  α={alpha_val:.3f}  p_KS={pval_mc:.3f}")
        print(f"{header}")
        print(f"{sep}")
    
        deg_filt = deg_arr[deg_arr > 0].astype(float)
        lrt = likelihoodRatioTests(deg_filt, kmin_val, alpha_val)
    
        for alt_name, (LR, p) in lrt.items():
            if np.isnan(p):
                msg = "No calculable"
            elif p >= 0.1:
                msg = "No distinguibles"
            elif LR > 0:
                msg = "Ley de potencias favorable"
            else:
                msg = "Alternativa favorable"
    
            bold_p = "**" if (not np.isnan(p) and p < 0.1) else "  "
            print(f"{alt_name:<18} {LR:>10.3f} {bold_p}{p:>6.3f}  {msg}")
    
file_path = "pypi_multiseed_10k.graphml"
G = ig.Graph.Read_GraphML(file_path)
G_giant, G_weak, G_strong, _, _, _, _, _, _, _, _ = graphStatistics(G)
islands = isolatedComponents(G)
_ = findCycle(G)
results = freescaleTests(G)
powerLawTest(G)
    
    




        
    