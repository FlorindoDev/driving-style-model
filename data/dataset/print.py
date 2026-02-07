import numpy as np

path = "normalized_dataset2.npz"  # cambia con il tuo file
d = np.load(path, allow_pickle=True)

print("Keys:", d.files)
for k in d.files:
    a = d[k]
    print(f"\n--- {k} ---")
    print("type:", type(a))
    print("dtype:", a.dtype)
    print("shape:", a.shape)

    # Se è un array object (tipico per sequenze variabili), mostra il contenuto del primo elemento
    if a.dtype == object:
        print("object array length:", len(a))
        x0 = a[0]
        print("first element type:", type(x0))
        try:
            x0 = np.asarray(x0)
            print("first element shape:", x0.shape)
            print("first element dtype:", x0.dtype)
            # attenzione: se x0 è scalare non puoi fare x0[:3]
            if x0.ndim >= 1:
                print("first element preview:\n", x0[:3])
            else:
                print("first element value:", x0)
        except Exception as e:
            print("cannot convert first element to ndarray:", e)
    else:
        # preview dei primi valori (attenzione se è enorme)
        flat = a.ravel()
        print("preview:", flat[:10])

# ---- STAMPA UNA RIGA DI data E mask ----
if "data" in d.files and "mask" in d.files:
    data = d["data"]
    mask = d["mask"]

    row_idx = 0  # cambia indice riga qui

    # stampa parziale per non floodare console
    print(f"\n=== ROW {row_idx} (first 30 values) ===")
    print("data :", data[row_idx, :50])
    print("mask :", mask[row_idx, :50])
    print("valid count:", int(mask[row_idx].sum()), "/", mask.shape[1])

    # se vuoi tutta la riga (455 valori), sblocca queste 2 righe:
    # np.set_printoptions(threshold=np.inf, linewidth=200)
    # print("data FULL:", data[row_idx]); print("mask FULL:", mask[row_idx])
else:
    print("\nNel file non esistono entrambe le chiavi 'data' e 'mask'.")
