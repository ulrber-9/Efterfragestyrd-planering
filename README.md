# Efterfragestyrd-planering

Detta projekt innehåller en Python-skriptfil som:

- Laddar in busstrafikdata (biljettvalideringar & realtidsdata).
- Om OSM-data saknas, hämtar och bearbetar den relevant OSM-information.
- Matchar på- och avstigningar för att skapa en OD-matris (Origin–Destination).

# Beskrivning av koden

## Vad koden gör

1. **Laddar in data (steg 1)**  
   - Hämtar nödvändiga CSV-filer med:
     - **TicketValidations** (biljettvalideringar)
     - **Realtidsdata** (bussars ankomst- och avgångstider)
     - **StopKey** (nyckeldata för busshållplatser)

2. **Kontrollerar och kör OSM-datahämtning (steg 2)**  
   - Ser efter om redan existerande OSM-datafiler (vägar, hållplatser, restidsmatris m.m.) finns sparade.  
   - Om saknas, kör funktionen `bs.osm_data_run` för att hämta och bearbeta OpenStreetMap-data för vald kommun.

3. **Läser in restidsmatris (StopMatrix) (steg 3)**  
   - Använder filen `<kommunnamn>_bus_stop_travel_times.csv` som innehåller restider och avstånd mellan hållplatser.  
   - Denna kan vara förgenererad eller skapas i steg 2.

4. **Förbehandlar realtidsdata (steg 4)**  
   - Rensar och kompletterar data: konverterar klockslag, fyller i saknade tider och skapar en kolumn för hållplatsordning (`StopSequence`).

5. **Förbehandlar biljettvalideringar (steg 5)**  
   - Rensar bort ogiltiga resenärs-ID, sorterar i tidsordning och ger varje rad ett unikt ID-nummer (`obs`).

6. **Skapar möjliga kombinationer av på- och avstigning (steg 6)**  
   - Bygger en “stop combination”-matris ur realtidsdatan för att koppla varje tänkbar start-/sluthållplats på samma resa.

7. **Kopplar biljettvalideringar till nästa möjliga hållplats (steg 7 & 8)**  
   - Matchar var resenären kan ha klivit av genom att jämföra biljettvalideringarna med restidsmatrisen och faktiska bussrutter.

8. **Beräknar “headways” (steg 10)**  
   - Räknar hur ofta (i tid) bussarna går från en viss hållplats för att avgöra om byten är rimliga.

9. **Bestämmer slutgiltig avstigningshållplats (steg 11)**  
   - Använder headways för att avgöra om resenären klivit av direkt eller bytt buss och fortsatt samma dag.

10. **Skapar en OD-matris (Origin-Destination) (steg 12)**  
   - Räknar hur många resenärer som rest mellan olika hållplatser per dag.

11. **(Valfritt) Visualiserar data (steg 13)**  
   - Om `Plot_Check` eller `plot_bool` är True, plottas bland annat vägnät, hållplatser, byggnader, vattenvägar och OD-linjer.

---

## Vilken indata behövs?

- **TicketValidations.csv**  
  Innehåller biljettvalideringar (resenärs-ID, datum, tid, linjenummer, hållplatsnummer m.m.).

- **Realtidsdata.csv**  
  Realtidsinformation om bussars avgångar och ankomster, t.ex. planerad och faktisk avgångstid, linjenummer, hållplatsnummer.

- **StopKey.csv**  
  En nyckelfil över busshållplatser (ID, namn, koordinater i Sweref99 TM).

- **OSM-datafiler**  
  - `<kommunnamn>_bus_stop_travel_times.csv` (restidsmatris mellan hållplatser)  
  - `<kommunnamn>_osm_data.pkl` (pickle-fil med vägnät, byggnader, vattenvägar och busshållplatser)

> Om dessa OSM-filer inte finns, försöker koden automatiskt generera dem via `bs.osm_data_run()` (OSM_BUS_STOP_PATHS.py).

> **Notera:** Du kan behöva uppdatera sökvägar i koden (t.ex. `mainPath`) och namnet på kommunen (`municipality_name`) för att matcha dina lokala filer.

---

## Vilken utdata skapas?

- **`Output_OD_Matrix.csv`**  
  En OD-matris (Origin–Destination) per dag. Varje rad visar:
  - `ValidationDate` – Datum för resorna  
  - `BoardingStop` – Ursprungshållplats  
  - `Final_AlightingStop` – Sluthållplats (efter ev. byten)  
  - `count` – Antal resenärer

- **(Eventuell) Visualisering**  
  Om `Plot_Check` är True genereras en karta som visar vägnät, hållplatser och OD-linjer (med linjebredd beroende på resenärsflöde).

- **Mellanfiler (debug)**  
  Exempelvis `Itinerary.csv` eller `AlightingStop.csv` om du aktiverar vissa `to_csv()`-anrop i koden. Dessa är dock inte nödvändiga i huvudflödet.

---

## Sammanfattning

Kort sagt utför skriptet en helhetsanalys för busstrafik från rå data (biljettvalideringar och realtidsinformation) till en färdig OD-matris. Du får möjlighet att se hur många som rest mellan olika hållplatser per dag och även, om så önskas, en grafisk plot över nätverket och resandeströmmar.

