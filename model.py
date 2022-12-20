from constants import s_label, j_label, l_label, g_label, i_label

class Model:
    def __init__(
        self,
        s, j, l, g, i,
        kjs, kgj, kjl, kgl, kxg, kxgi, kxi,
        tau, klambda, mu, beta, gamma,
        gprod0, gb, ib,
        fgj
    ):
        self.history = []
        self.history.append({'step': 0, s_label: s, j_label: j, l_label: l, g_label: g, i_label: i})

        self.kjs = kjs
        self.kgj = kgj
        self.kjl = kjl
        self.kgl = kgl
        self.kxg = kxg
        self.kxgi = kxgi
        self.kxi = kxi

        self.tau = tau
        self.klambda = klambda
        self.mu = mu
        self.beta = beta
        self.gamma = gamma

        self.gprod0 = gprod0
        self.gb = gb
        self.ib = ib

        self.fgj = fgj


    def update(self, t):
        old_s = self.history[t - 1][s_label]
        old_j = self.history[t - 1][j_label]
        old_l = self.history[t - 1][l_label]
        old_g = self.history[t - 1][g_label]
        old_i = self.history[t - 1][i_label]

        new_s = old_s - (old_s * self.kjs)
        new_j = old_j + (old_s * self.kjs) - (old_j * self.kgj) - (old_j * self.kjl)

        phi = 0 if t < self.tau else self.history[t - self.tau][j_label]
        new_l = old_l + (phi * self.kjl) - (old_l * self.kgl)

        g_prod = self.klambda / ((self.klambda / self.gprod0) + (old_g - self.gb))
        new_g = old_g - (self.kxg + self.kxgi * old_i) * old_g + g_prod + self.mu * (self.kgj * old_j + self.kgl * old_l)
        
        g_tilde = old_g + self.fgj * (self.kgj * old_j + self.kgl * old_l)
        new_i = old_i + self.kxi * self.ib * ((self.beta ** self.gamma + 1) / (self.beta ** self.gamma * (self.gb / g_tilde) ** self.gamma + 1) - old_i / self.ib)

        timestep = {'step': t, s_label: new_s, j_label: new_j, l_label: new_l, g_label: new_g, i_label: new_i}
        self.history.append(timestep)