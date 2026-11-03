import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from scipy.stats import chi2 as chi2_dist
except ImportError:  # scipy необов'язковий
    chi2_dist = None

# Константи
NAMES = ["Стандартний генератор", "Метод середини квадрату", "Лінійний конгруентний метод"]
SHEETS = ["Стандартний", "Середина квадрату", "ЛКМ"]
PARAM_SHEET = "Параметри"
MAX_LIST = 2000        # скільки чисел показувати у списку
MAX_TRIALS = 1_000_000


# Генератори
# Усі повертають цілі числа з відрізка [0; n]
def standard_generator(n, count):
    """Стандартний генератор (вбудований random)."""
    return [random.randint(0, n) for _ in range(count)]


def middle_square_generator(n, count, seed=8219):
    """
    Метод середини квадрату.
    x(i+1) = середні 4 цифри від x(i)^2 (квадрат доповнюється нулями до 8 цифр).
    y = x / 10^4 лежить у [0; 1), далі масштабується на [0; n].
    Якщо послідовність вироджується (x = 0) або зациклюється,
    береться нове початкове значення, а кількість таких випадків повертається.
    """
    x = seed
    seen = {x}
    result = []
    reseeds = 0
    for _ in range(count):
        x = (x * x // 100) % 10000
        if x == 0 or x in seen:
            reseeds += 1
            x = random.randint(1000, 9999)
            seen = set()
        seen.add(x)
        result.append(int(x / 10000 * (n + 1)))
    return result, reseeds


def linear_congruential_generator(n, count, k=1103515245, c=12345,
                                  modulus=2 ** 31, seed=1):
    """
    Лінійний конгруентний метод.
    x(i+1) = (k * x(i) + c) mod modulus,  y = x / modulus  лежить у [0; 1).
    """
    x = seed
    result = []
    for _ in range(count):
        x = (k * x + c) % modulus
        result.append(int(x / modulus * (n + 1)))
    return result


# Підрахунок та статистика
def count_hits(values, n, m):
    """Кількість «попадань» у кожен з m рівних проміжків відрізка [0; n]."""
    h = n / m
    counts = [0] * m
    for z in values:
        counts[min(int(z / h), m - 1)] += 1
    return counts


def interval_labels(n, m):
    h = n / m
    labels = []
    for i in range(m):
        a, b = i * h, (i + 1) * h
        fa = f"{a:g}"
        fb = f"{b:g}"
        labels.append(f"[{fa};{fb}{']' if i == m - 1 else ')'}")
    return labels


def sample_statistics(values, counts, n, m):
    """Характеристики вибірки та хі-квадрат відносно рівномірного розподілу."""
    arr = np.array(values, dtype=float)
    total = len(arr)
    expected = total / m
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    stats = {
        "Середнє": arr.mean(),
        "Дисперсія": arr.var(),
        "СКВ (σ)": arr.std(),
        "Мінімум": arr.min(),
        "Максимум": arr.max(),
        "Медіана": float(np.median(arr)),
        "χ²": chi2,
    }
    if chi2_dist is not None and m > 1:
        stats["χ² крит. (0.95)"] = float(chi2_dist.ppf(0.95, m - 1))
    return stats


# Графічний інтерфейс
class App:
    def __init__(self, root):
        self.root = root
        root.title("Імовірнісне моделювання: метод Монте-Карло")
        root.geometry("1180x860")

        self.data = [[], [], []]     # згенеровані числа для кожного генератора
        self.reseeds = 0
        self.figs, self.axes, self.canvases, self.listboxes = [], [], [], []

        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        right = ttk.Frame(root, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        # --- три блоки: діаграма + список чисел ---
        for i, name in enumerate(NAMES):
            box = ttk.LabelFrame(left, text=name)
            box.pack(fill=tk.BOTH, expand=True, pady=3)

            fig = Figure(figsize=(6.2, 2.3), dpi=90)
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, master=box)
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            lb_frame = ttk.Frame(box)
            lb_frame.pack(side=tk.RIGHT, fill=tk.Y)
            ttk.Label(lb_frame, text="Отримані числа").pack()
            lb = tk.Listbox(lb_frame, width=12, height=8)
            sb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL, command=lb.yview)
            lb.config(yscrollcommand=sb.set)
            lb.pack(side=tk.LEFT, fill=tk.Y)
            sb.pack(side=tk.LEFT, fill=tk.Y)

            self.figs.append(fig)
            self.axes.append(ax)
            self.canvases.append(canvas)
            self.listboxes.append(lb)
            self._draw_empty(i)

        # --- панель керування ---
        ttk.Label(right, text="Інтервал генерації чисел\nвід 0 до").pack(pady=(4, 0))
        self.var_n = tk.StringVar(value="100")
        ttk.Entry(right, textvariable=self.var_n, width=12, justify="center").pack()

        ttk.Separator(right).pack(fill=tk.X, pady=6)
        ttk.Label(right, text="Кількість проміжків").pack()
        self.var_m = tk.StringVar(value="5")
        ttk.Entry(right, textvariable=self.var_m, width=12, justify="center").pack()

        ttk.Separator(right).pack(fill=tk.X, pady=6)
        ttk.Label(right, text="Кількість випробувань").pack()
        self.var_N = tk.StringVar(value="50")
        ttk.Entry(right, textvariable=self.var_N, width=12, justify="center").pack()

        ttk.Separator(right).pack(fill=tk.X, pady=6)
        adv = ttk.LabelFrame(right, text="Параметри методів")
        adv.pack(fill=tk.X)
        self.var_ms_seed = tk.StringVar(value="8219")
        self.var_k = tk.StringVar(value="1103515245")
        self.var_c = tk.StringVar(value="12345")
        self.var_mod = tk.StringVar(value=str(2 ** 31))
        self.var_lcg_seed = tk.StringVar(value="1")
        rows = [("x0 (сер. квадрату, 4 цифри)", self.var_ms_seed),
                ("Множник k", self.var_k),
                ("Зсув c", self.var_c),
                ("Модуль", self.var_mod),
                ("x0 (ЛКМ)", self.var_lcg_seed)]
        for text, var in rows:
            ttk.Label(adv, text=text).pack(anchor="w", padx=4)
            ttk.Entry(adv, textvariable=var, width=22).pack(padx=4, pady=(0, 3))

        ttk.Separator(right).pack(fill=tk.X, pady=6)
        ttk.Button(right, text="Згенерувати", command=self.generate).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Експортувати в Excel", command=self.export_excel).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Імпортувати з Excel", command=self.import_excel).pack(fill=tk.X, pady=2)

        ttk.Label(right, text="Характеристики вибірки").pack(pady=(8, 0))
        self.txt = tk.Text(right, width=38, height=22, font=("Consolas", 9), state=tk.DISABLED)
        self.txt.pack(fill=tk.BOTH, expand=True)

    # Допоміжні методи
    def _draw_empty(self, i):
        ax = self.axes[i]
        ax.clear()
        ax.set_title(NAMES[i], fontsize=9, fontstyle="italic", fontweight="bold")
        self.figs[i].tight_layout()
        self.canvases[i].draw()

    def _read_params(self):
        """Зчитує та перевіряє параметри. При помилці повертає None."""
        try:
            n = int(self.var_n.get())
            m = int(self.var_m.get())
            N = int(self.var_N.get())
            ms_seed = int(self.var_ms_seed.get())
            k = int(self.var_k.get())
            c = int(self.var_c.get())
            modulus = int(self.var_mod.get())
            lcg_seed = int(self.var_lcg_seed.get())
        except ValueError:
            messagebox.showerror("Помилка", "Усі поля мають містити цілі числа.")
            return None

        errors = []
        if n < 1:
            errors.append("Кінець інтервалу n має бути ≥ 1.")
        if m < 1 or m > 100:
            errors.append("Кількість проміжків має бути від 1 до 100.")
        if N < 1 or N > MAX_TRIALS:
            errors.append(f"Кількість випробувань має бути від 1 до {MAX_TRIALS}.")
        if not 1000 <= ms_seed <= 9999:
            errors.append("Початкове число методу середини квадрату має бути чотиризначним.")
        if k < 1 or c < 0 or modulus < 2 or lcg_seed < 0:
            errors.append("Параметри ЛКМ: k ≥ 1, c ≥ 0, модуль ≥ 2, x0 ≥ 0.")
        if errors:
            messagebox.showerror("Помилка введення", "\n".join(errors))
            return None
        return n, m, N, ms_seed, k, c, modulus, lcg_seed

    def _show_results(self, n, m):
        """Малює діаграми, заповнює списки та поле зі статистикою."""
        labels = interval_labels(n, m)
        lines = []
        for i in range(3):
            values = self.data[i]
            counts = count_hits(values, n, m)

            ax = self.axes[i]
            ax.clear()
            bars = ax.bar(range(m), counts, color="royalblue", edgecolor="navy")
            ax.bar_label(bars, fontsize=8)
            ax.set_xticks(range(m))
            ax.set_xticklabels(labels, fontsize=7, rotation=0 if m <= 8 else 60)
            ax.set_title(NAMES[i], fontsize=9, fontstyle="italic", fontweight="bold")
            ax.set_ylim(0, max(counts) * 1.2 if max(counts) else 1)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            self.figs[i].tight_layout()
            self.canvases[i].draw()

            lb = self.listboxes[i]
            lb.delete(0, tk.END)
            for v in values[:MAX_LIST]:
                lb.insert(tk.END, v)
            if len(values) > MAX_LIST:
                lb.insert(tk.END, f"... (показано {MAX_LIST})")

            st = sample_statistics(values, counts, n, m)
            lines.append(f"== {NAMES[i]} ==")
            lines.append("Попадання: " + ", ".join(map(str, counts)))
            for key, val in st.items():
                lines.append(f"{key:<16}{val:>12.4f}")
            lines.append("")

        N = len(self.data[0])
        theor_mean = n / 2
        theor_var = ((n + 1) ** 2 - 1) / 12
        lines.append("Теорія (рівномірний):")
        lines.append(f"Середнє         {theor_mean:>12.4f}")
        lines.append(f"Дисперсія       {theor_var:>12.4f}")
        lines.append(f"Очікувано в проміжку ≈ {N / m:.2f}")
        if self.reseeds:
            lines.append("")
            lines.append(f"Сер. квадрату: виродження/зациклення, нових x0: {self.reseeds}")

        self.txt.config(state=tk.NORMAL)
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "\n".join(lines))
        self.txt.config(state=tk.DISABLED)

    # Кнопки
    def generate(self):
        params = self._read_params()
        if params is None:
            return
        n, m, N, ms_seed, k, c, modulus, lcg_seed = params

        std = standard_generator(n, N)
        ms, self.reseeds = middle_square_generator(n, N, ms_seed)
        lcg = linear_congruential_generator(n, N, k, c, modulus, lcg_seed)
        self.data = [std, ms, lcg]
        self._show_results(n, m)

    def export_excel(self):
        if not self.data[0]:
            messagebox.showwarning("Увага", "Спочатку натисніть «Згенерувати».")
            return
        params = self._read_params()
        if params is None:
            return
        n, m = params[0], params[1]

        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")],
            initialfile="monte_carlo.xlsx")
        if not path:
            return
        try:
            labels = interval_labels(n, m)
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pd.DataFrame({"Параметр": ["n", "m", "N"],
                              "Значення": [n, m, len(self.data[0])]}
                             ).to_excel(writer, sheet_name=PARAM_SHEET, index=False)
                for sheet, values in zip(SHEETS, self.data):
                    pd.DataFrame({"№": range(1, len(values) + 1),
                                  "Число": values}
                                 ).to_excel(writer, sheet_name=sheet, index=False)
                    pd.DataFrame({"Проміжок": labels,
                                  "Кількість": count_hits(values, n, m)}
                                 ).to_excel(writer, sheet_name=sheet, index=False, startcol=3)
            messagebox.showinfo("Готово", f"Дані збережено:\n{path}")
        except Exception as e:
            messagebox.showerror("Помилка експорту", str(e))

    def import_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls")])
        if not path:
            return
        try:
            book = pd.read_excel(path, sheet_name=None)
            data = []
            for sheet in SHEETS:
                if sheet not in book or "Число" not in book[sheet].columns:
                    raise ValueError(f"У файлі немає аркуша «{sheet}» зі стовпцем «Число».")
                data.append([int(v) for v in book[sheet]["Число"].dropna()])
            if not data[0]:
                raise ValueError("Файл не містить даних.")

            n, m = int(self.var_n.get()), int(self.var_m.get())
            if PARAM_SHEET in book:
                p = dict(zip(book[PARAM_SHEET]["Параметр"], book[PARAM_SHEET]["Значення"]))
                n, m = int(p.get("n", n)), int(p.get("m", m))
            self.var_n.set(str(n))
            self.var_m.set(str(m))
            self.var_N.set(str(len(data[0])))

            self.data = data
            self.reseeds = 0
            self._show_results(n, m)
        except Exception as e:
            messagebox.showerror("Помилка імпорту", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()