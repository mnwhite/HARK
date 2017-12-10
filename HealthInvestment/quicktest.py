for k in range(14):
    bot = 720 + k*75
    top = bot + 75
    Z = np.linalg.inv(CovMatrix[bot:top,bot:top])
    plt.plot(np.dot(Z,CovMatrix[bot:top,bot:top]).flatten())
    plt.show()
