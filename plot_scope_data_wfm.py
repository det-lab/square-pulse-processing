import matplotlib.pyplot as plt
import RigolWFM.wfm as rigol

def load_and_plot(filename="data/NewFile1.wfm"):
    # Read the CSV file
    scope_data = rigol.Wfm.from_file(filename, '1000Z')

    # Plot the data
    plt.figure(figsize=(12, 5))
    for ch in scope_data.channels:
        print(ch)
        plt.plot(ch.times, ch.volts, label=ch.name)
    plt.legend()
    plt.xlabel("Time (ns)")
    plt.ylabel("Voltage (V)")
    plt.title(filename)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


load_and_plot("data/NewFile2.wfm")

print("Done.")