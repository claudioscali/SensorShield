import torch
import torch.nn as nn

class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim=140): #ogni onda temporale del nostro dataset è formata da 140 misurazioni consecutive
        super(AnomalyAutoencoder, self).__init__()
        
        # ENCODER: Comprime i dati. Da 140 punti (l'onda) a soli 8 numeri.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), # Uno strato Fully Connected. Prende i 140 numeri in ingresso e li moltiplica per una matrice di pesi interni per restituire 32 numeri
            nn.ReLU(True), # prende i risultati del livello precedente, lascia invariati i numeri positivi e trasforma tutti i numeri negativi in 0. Non-linearità => imparare forme complesse (curve delle onde).
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8), # 8 numeri finali sono il bottleneck o spazio latente, 8 coordinate essenziali
            nn.ReLU(True)
        )
        
        # DECODER: Prova a ricostruire l'onda originale partendo dai 8 numeri.
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        # Il flusso dei dati: entra, si restringe, si espande, ed esce
        x = self.encoder(x)
        x = self.decoder(x)
        return x