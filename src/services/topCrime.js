export const getTopCrimeCity = (crimeData) => {
  const { labels, datasets } = crimeData;
  const crimeCounts = datasets[0].data;

  const citiesWithCrimes = labels.map((city, index) => ({
    city,
    crimeCount: crimeCounts[index],
  }));

  const sortedCities = citiesWithCrimes.sort(
    (a, b) => b.crimeCount - a.crimeCount
  );

  return sortedCities.slice(0, 5);
};
