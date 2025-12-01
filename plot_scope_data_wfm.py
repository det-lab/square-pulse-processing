import matplotlib.pyplot as plt
import RigolWFM.wfm as rigol

def load_and_plot(filename):
    # Read the CSV file
    scope_data = rigol.Wfm.from_file(filename, '1000Z')

    # Plot the data
    plt.figure(figsize=(12, 5))
    for ch in scope_data.channels:
        print(ch)
        plt.plot(ch.times, ch.volts, label=ch.name)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(filename)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


load_and_plot("C:\\Users\\canto\\Data\\500_micro_s\\sample_3.wfm")

print("Done.")