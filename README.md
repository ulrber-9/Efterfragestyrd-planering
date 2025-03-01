# Efterfragestyrd-planering

Detta projekt innehåller en Python-skriptfil som:

- Laddar in busstrafikdata (biljettvalideringar & realtidsdata).
- Om Open Street Map (OSM-) data saknas, hämtar och bearbetar relevant OSM-information.
- Matchar på- och avstigningar för att skapa en OD-matris (Origin–Destination).

Kort sagt utför skriptet en helhetsanalys för busstrafik från rå data (biljettvalideringar och realtidsinformation) till en färdig OD-matris. Du får möjlighet att se hur många som rest mellan olika hållplatser per dag.

# Beskrivning av koden

## Vilken indata behövs?

- Du behöver uppdatera sökvägar i koden (t.ex. `mainPath`) och namnet på kommunen (`municipality_name`) för att matcha dina lokala filer. Det finns också ett antal andra parametrar som kan ändras.

- **TicketValidations.csv**  
  Innehåller biljettvalideringar (resenärs-ID, datum, tid, linjenummer, hållplatsnummer m.m.).

- **Realtidsdata.csv**  
  Realtidsinformation om bussars avgångar och ankomster, t.ex. planerad och faktisk avgångstid, linjenummer, hållplatsnummer.

- **StopKey.csv**  
  En nyckelfil över busshållplatser (ID, namn, koordinater i Sweref99 TM).
  
> Koden förväntar sig csv filer med komma ( , ) som avgränsare och punkt ( . ) för decimaler
> Koden förväntar sig också att csv filerna har "encoding='utf-8-sig'"

- **OSM-datafiler**  
  - `<kommunnamn>_bus_stop_travel_times.csv` (restidsmatris mellan hållplatser)  
  - `<kommunnamn>_osm_data.pkl` (pickle-fil med vägnät, byggnader, vattenvägar och busshållplatser)

> Om dessa OSM-filer inte finns, försöker koden automatiskt generera dem via `bs.osm_data_run()` som finns i `OSM_BUS_STOP_PATHS.py` skriptet.

---

## Vilken utdata skapas?

- **`Output_OD_Matrix.csv`**  
  En OD-matris (Origin–Destination) per dag. Varje rad visar:
  - `ValidationDate` – Datum för resorna  
  - `BoardingStop` – Ursprungshållplats  
  - `Final_AlightingStop` – Sluthållplats (efter ev. byten)  
  - `count` – Antal resenärer

- **(Eventuell) Visualisering**  
  Export_to_SHP skriptet exportar OD matrisen ifrån Excel till en shapefile samt skapar en visuell karta som kan öppnas i Google Chrome

- **Mellanfiler (debug)**  
  Exempelvis `Itinerary.csv` eller `AlightingStop.csv` om du aktiverar vissa `to_csv()`-anrop i koden. Dessa är dock inte nödvändiga i huvudflödet.

---

## Vad koden gör

1. **Laddar in data (steg 1)**  
   - Hämtar nödvändiga CSV-filer med:
     - **TicketValidations** (biljettvalideringar)
     - **Realtidsdata** (bussars ankomst- och avgångstider)
     - **StopKey** (nyckeldata för busshållplatser)

> Koden förväntar sig csv filer med komma ( , ) som avgränsare och punkt ( . ) för decimaler

2. **Kontrollerar och kör OSM-datahämtning (steg 2)**  
   - Ser efter om redan existerande OSM-datafiler (vägar, hållplatser, restidsmatris m.m.) finns sparade.  
   - Om saknas, kör funktionen `bs.osm_data_run` för att hämta och bearbeta OpenStreetMap-data för vald kommun.

3. **Läser in restidsmatris (StopMatrix) (steg 3)**  
   - Använder filen `<kommunnamn>_bus_stop_travel_times.csv` som innehåller restider och avstånd mellan hållplatser.  
   - Skapas i steg 2.

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
   - Om `Plot_Check` är True, plottas bland annat vägnät, hållplatser, byggnader, vattenvägar och OD-linjer.
   - Det finns också ett Export_to_SHP skript med fler visualiseraringar samt export till shapefil

---

## Hur kör man koden

>För att koden ska fungera behövs Python version 3.10+
>Du måste dessutom installera ett antal bibliotek. Använd kommandot "pip install -r requirements.txt" i kommandotolken för detta!
>Se till att alla csv filer har rätt "encoding". Enklast görs detta genom att spara data som .csv ifrån Excel


