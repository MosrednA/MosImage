# MosImage: modelonderzoek en productrichting

Status: Historical
Last verified: 2026-10-11
Current state: [Guide](../guide.md) and [roadmap](../roadmap.md).

Onderzocht op **7 oktober 2026**. Status: **onderzoekssnapshot en voorstel**, met inmiddels deels uitgevoerde aanbevelingen. De beginsituatie hieronder beschrijft de inspectie voor implementatie. De huidige functies staan in de [guide](../guide.md); [Focus-implementatie en validatie](2026-10-07-focus-implementation.md) bewaart de leveringsgeschiedenis en de [roadmap](../roadmap.md) beschrijft resterend werk.

De regionale Klein-route is daarna vervangen door native mask conditioning, met een zichtbare contextrechthoek en instelbare blending. De toenmalige route behield zachte maskerwaarden tijdens sampling en gebruikte 320 px context en 16 px inward blend. Sinds 9–10 oktober gebruikt de productiecode aparte generation- en blendmaskers, automatisch ingestelde edges en verborgen generation-overlap; de guide beschrijft die actuele instellingen. De oorspronkelijke observatie dat Klein het masker negeerde blijft hieronder bewaard als beginsituatie; [inpainting-implementatie en proeven](2026-10-07-inpainting.md) bewaart die implementatie en negatieve referentie-experimenten.

## Advies

Maak MosImage een lokale, visuele beeldwerkplek: **open of maak een beeld, wijs aan wat moet veranderen, bekijk voorstellen en pas er één toe**. Houd het canvas centraal. Modellen, maskers en geheugenplaatsing ondersteunen die handeling.

De huidige technische basis verdient behoud. De grootste productwinst komt van objectselectie, canvas uitbreiden, betere maskergestuurde bewerking en minder zichtbare bediening. Nieuwe modellen moeten eerst tegen die taken worden gemeten.

## Beginsituatie bij het onderzoek

Inspectie van de broncode, package metadata en de frontend:

- React 19.2.7, Vite 7.3.5, TypeScript en een Tauri 2 scaffold.
- FastAPI 0.136.3, SQLite, een seriële GPU-worker en cooperatieve annulering met supervisor-herstart.
- PyTorch 2.11.0+cu128, Transformers 5.11.0 en Diffusers 0.40.0.dev0. Diffusers is lokaal afkomstig van commit `f6a3517673e32b4134cd4eb62bd13a20934385eb`, maar requirements installeren de bewegende Git-bron zonder die commit te pinnen.
- FLUX.2 Klein 4B als Quality, Z-Image Turbo als Fast, optioneel FLUX.1 Fill Dev.
- Een canvas met beeldcoordinaten, handmatige selecties, maskerbewerkingen, documenten, revisies, alternatieven en native-resolutie compositing.
- Nog geen SAM-objectselectie, outpaint-route of referentiebeeldbeheer. `TaskKind` ondersteunt momenteel alleen text-to-image, inpaint en load-model.
- De Klein-provider negeert `mask_path` expliciet en bewerkt de contextcrop op basis van de instructie. De compositor beperkt vervolgens welke pixels terugkomen. Dat is een nuttige beschermingsgarantie, maar het model ontvangt niet de ruimtelijke selectie.
- De compositor zet bron en gegenereerd beeld om naar RGB. Transparante documenten zijn daarmee nog geen end-to-end ondersteunde workflow.
- De UI heeft aparte Generate/Edit-modi en een permanent instellingenpaneel van 360 px. Prompt, kwaliteit, afmetingen, alternatieven, technische instellingen, recente resultaten en documenten concurreren om diezelfde kolom.
- Er zijn inmiddels backend-unittests en Vitest-tests. De algemene AGENTS.md-beschrijving dat er nog geen test suite is, loopt achter op de bestanden.

De frontend is tijdelijk gestart voor scherminspectie. De backend draaide tijdens deze inspectie niet; de zichtbare fetch-fout is daarom geen vastgestelde runtimebug. Er zijn geen nieuwe modellen gedownload en er zijn geen GPU-kwaliteits- of snelheidsbenchmarks uitgevoerd.

## Modelkeuzes, met hun werkelijke rol

| Model / techniek | Relevant voor | Advies voor MosImage |
|---|---|---|
| FLUX.2 Klein 4B | Generatie, instructiebewerking en meerdere referenties; Apache 2.0 | Bestaande lokale standaard behouden. Referenties ontsluiten en maskergestuurde bewerking onderzoeken. |
| Z-Image Turbo | Compacte, gedistilleerde generatie | Behouden als tweede renderstijl zolang een lokale vergelijking zijn toegevoegde waarde laat zien. |
| Qwen-Image 2.1 | Nieuw verenigd generatie/editmodel, RGBA, referenties en lokaal aangeduide edits | Belangrijkste nieuwe evaluatiekandidaat; geen automatische vervanging. |
| Qwen-Image-Edit-2511 | Referentie- en instructiebewerking met Apache 2.0 | Vergelijkingskandidaat als licentie van 2.1 niet past; zwaardere 20B-klasse. |
| FLUX.1 Fill Dev | Specialist met native image-plus-mask invoer | Bestaande optionele specialist en vergelijkingsbaseline behouden. |
| FLUX.2 Klein 9B / 9B KV / Dev | Grotere lokale modellen; KV-variant gericht op referentiebewerking | Later evalueren, vanwege geheugen en andere licentievoorwaarden. |
| Krea 2 Turbo / RAW | Esthetische generatie; Turbo voor inference, RAW voor tuning | Optionele creatieve richting. Geen directe oplossing voor alle editproblemen. |
| SAM 3 | Object- en conceptselecties via klik, box en tekst | Hoogste prioriteit voor een nieuwe ondersteunende modelprovider. |
| LanPaint 2.2 | Samplingmethode voor maskergestuurde inpaint/outpaint zonder apart fill-checkpoint | Afzonderlijke technische proef; niet behandelen als een drop-in model. |
| GPT-Image-2.5 Flare / Sunburst | Cloudgeneratie en inpainting; snelheids- versus precisierol | Optionele toekomstige kwaliteitsvergelijking of expliciet ingeschakelde cloudprovider. |
| Nano Banana 2.1 / Pro | Cloudbewerking met referenties en opeenvolgende instructies | Dezelfde optionele cloudrol; lokale werking blijft de standaard. |

### Bronnen en grenzen

**FLUX.2 Klein:** de officiele vergelijking noemt ongeveer 8.4 GB VRAM voor de gedistilleerde 4B, tegenover 19.6 GB voor 9B. Andere BFL-hulppagina's geven hogere praktijkramingen. Dit zijn providerprofielen, geen metingen op de RTX 5070 Ti. Gebruik onze eigen huidige offload-route als uitgangspunt en meet dezelfde resolutie, referenties en runtime. 4B gebruikt Apache 2.0; de grotere varianten gebruiken andere voorwaarden. [BFL-modeloverzicht](https://bfl.ai/models/flux-2-klein), [officiele inference-repository](https://github.com/black-forest-labs/flux2).

**Qwen-Image 2.1:** gepubliceerd op 20 september 2026. De modelcard noemt een 7B visuele component, tot tien referentiebeelden, aangegeven lokale edits en transparante generatie. Die 7B beschrijft niet de hele pipeline: alleen de gepubliceerde text-encoderbestanden beslaan al 17.5 GB op schijf. Volledige BF16-residentie op 16 GB is dus geen aannemelijke standaard; offload en/of quantisatie moeten worden gemeten. De quickstart vraagt Transformers >=5.17, hoger dan de lokaal gevonden 5.11.0. [Modelcard](https://huggingface.co/Qwen/Qwen-Image-2.1), [encoderbestanden](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/main/text_encoder).

Qwen 2.1 heeft een **Qwen Research License** met non-commercial onderzoeks/evaluatiegebruik en een aparte commerciele licentieroute. De oudere Edit-2511-card vermeldt Apache 2.0. Deze voorwaarden mogen niet door elkaar worden gehaald. [2.1-licentie](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), [Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).

**Z-Image:** Turbo blijft een 6B-generatiekandidaat. De officiele repository beschrijft Edit en Omni-Base, maar vermeldt die checkpoints nog als te publiceren; een aangekondigde editvariant is geen beschikbare dependency. [Officiele Z-Image-repository](https://github.com/Tongyi-MAI/Z-Image).

**Fill:** FLUX.1 Fill Dev is een 12B native fillmodel met image-plus-mask invoer. Het is gated en heeft een non-commercial modellicentie; de modelcard maakt apart melding van rechten op gegenereerde outputs. De bestaande compositor blijft nodig, ook omdat de modelcard mogelijke kleurverschuiving buiten de fillregio benoemt. [FLUX.1 Fill-card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev).

**Krea:** Krea 2 levert RAW en een 8-step Turbo voor text-to-image. De gewichten hebben een communitylicentie met voorwaarden voor deployment. De code-repositorylicentie beschrijft niet automatisch de rechten op de modelgewichten. De bestaande roadmap houdt Krea daarom terecht achter een gemeten 16 GB-route en licentiecontrole. [Krea-repository](https://github.com/krea-ai/krea-2), [Turbo-modelcard](https://huggingface.co/krea/Krea-2-Turbo).

**SAM / LanPaint:** SAM 3 maakt selecties, geen nieuwe beeldinhoud. LanPaint is een andere samplingroute; de officiele ComfyUI-implementatie verwijst ook naar een aparte Diffusers-integratie. Release 2.2.0 van 28 september voegt Qwen 2.1 toe. Dit bewijst geen integratiepariteit met MosImage. [SAM 3](https://ai.meta.com/research/sam3/), [LanPaint upstream](https://github.com/scraed/LanPaint), [releasehistorie](https://github.com/scraed/LanPaint/releases).

**Cloud:** de officiele OpenAI-documentatie vermeldt de op 8 september uitgebrachte GPT-Image-2.5 Flare en Sunburst, beide met inpainting. Google noemt inmiddels Nano Banana 2.1 als opvolger van Nano Banana 2, met Pro voor complexere visuele taken. Dit zijn API-routes en geen lokale checkpoints. Exacte kosten hangen af van invoer, uitvoer en kwaliteitsinstellingen; geen vast bedrag of accounttoegang is hier aangenomen. [Flare](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare), [Sunburst](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst), [Google image generation](https://ai.google.dev/gemini-api/docs/image-generation).

## Trends vertaald naar productfuncties

Dit is onze productinterpretatie van de bovenstaande mogelijkheden, geen bewezen kwaliteitsranglijst:

1. **Van genereren naar doorwerken.** Een beeld moet na elke edit zijn compositie en onderwerp behouden. Toon wijzigingen in context en laat de gebruiker elke stap terugdraaien.
2. **Taal plus aanwijzen.** Een korte instructie werkt beter als de gebruiker zichtbaar aangeeft waar die geldt. Combineer objectselectie, handmatige correctie en een contextgebonden prompt.
3. **Referenties als vaste houvast.** Een product, karakter of stijl krijgt een eigen referentieplaats, met een zichtbare rol: onderwerp, stijl of compositie. Behoud is een doel voor evaluatie, geen absolute modelgarantie.
4. **Selecteren wordt een eigen modeltaak.** Klikken of 'de achtergrond' typen moet een controleerbaar masker opleveren. Toon de selectie voordat een generatiejob start.
5. **Transparantie wordt een echte output.** Achtergrond verwijderen en een onderwerp als PNG bewaren zijn herkenbare taken. Eerst RGBA door documenten, previews, compositing en export laten lopen; niet alleen een RGBA-model toevoegen.
6. **Snelle voorstellen, zorgvuldige afwerking.** Laat het eerste alternatief verschijnen zodra het klaar is, terwijl de rest sequentieel doorloopt. Een snelle draft en een hoogwaardige eindrender zijn alleen dezelfde bewerking als het model dat daadwerkelijk ondersteunt.
7. **Controle boven een grotere modellijst.** Een klein betrouwbaar aanbod met meetbare editkwaliteit is bruikbaarder dan veel modellen waarvan capabilities en parameters onduidelijk zijn.

Andere nuttige vervolgtaken zijn achtergrond vervangen, selectieve materiaal-/kleurwijzigingen, relighting, schets/structuur als referentie en een aparte upscale/exportstap. Video, training en een volledige lageneditor blijven buiten de eerste iteratie.

## UX/UI: één werkplek

### Voorkeur: Focus

Een sober, ruim canvas met een smalle gelabelde gereedschapsrail. Onderaan staat een compacte opdrachtbalk. Bibliotheek, geschiedenis, referenties en technische instellingen openen alleen wanneer ze nodig zijn. De blauwe accenten sluiten aan bij de bestaande MosImage-identiteit; de selectiekleur blijft onderscheidbaar van de gewone bediening.

De kenmerkende interactie is dat **de opdracht bij de geselecteerde beeldinhoud hoort**. Er is één werkplek, met duidelijk gekozen acties; selectie alleen mag niet onverwacht de betekenis van een al geschreven opdracht veranderen.

- **Start:** 'Open afbeelding' en 'Maak afbeelding', plus recente documenten. Plakken en slepen werken zonder aparte modus.
- **Wijzigen:** klik een object of verf een regio. Een compact actiemenu biedt Vervangen, Verwijderen, Achtergrond en Selectie aanpassen.
- **Vervangen:** schrijf bijvoorbeeld 'Een houten huisje met een donkerblauw dak'. De actieknop heet 'Maak voorstellen'.
- **Verwijderen:** kan zonder eigen prompt; de actie maakt een passende verwijderinstructie. Waar nodig kan de gebruiker specificeren wat terug moet komen.
- **Uitbreiden:** sleep een canvasrand of kies 16:9. De nieuwe ruimte wordt geruit getoond. 'Vul nieuwe ruimte' is de concrete actie. Een automatische vervolginstructie is bruikbaar als geen tekst nodig is.
- **Beoordelen:** toon voorstellen in het canvas. 'Origineel tonen', de alternatievenstrip, 'Pas toe' en 'Verwerp' houden beoordeling overzichtelijk. Een gegenereerd voorstel is nog geen geaccepteerde documentwijziging.
- **Doorwerken:** accepted edits, undo/redo, documentnaam en autosave vormen de normale editorworkflow.
- **Exporteren:** formaat, afmetingen, transparantie en eventuele upscale staan bij export; een modelcrop van 1024 px is niet hetzelfde als een inhoudelijk nieuwe 4K-render.

De gebruiker koos vervolgens expliciet een **Engelse UI**. Gebruik Engelse termen consequent in de implementatie. Vermijd termen als denoise, CFG, inference, pipeline en VRAM in de normale bewerkingsflow. Bewaar technische informatie in een bereikbare detailweergave.

### Alternatief: Studio

Dezelfde werkplek met een blijvend, compact rechterpaneel voor instructie, referenties en bewerkingsgeschiedenis. Dit is bruikbaar bij lange werksessies en veel referenties. Het canvas krijgt minder ruimte. Focus is mijn aanbevolen default; de varianten zijn een ontwerpkeuze, geen verzoek om twee verschillende applicaties te bouwen.

### Interactiedetails die het verschil maken

- 'Sneller' en 'Meer detail' alleen gebruiken als de gekozen route die afweging ondersteunt. De huidige namen Fast/Quality beloven een rangorde die niet voor elke taak gemeten is.
- Geen creatieve-kracht-slider tonen voor een provider die die parameter negeert.
- Aanwijzingen voor 'wat verandert' en 'wat blijft' bij referenties en regionale edits.
- Space om te pannen, wielzoom rond de pointer, Ctrl+Z voor een voorspelbare laatste handeling, Escape voor selectie/voorstel annuleren.
- Selectiegeschiedenis en documentrevisies duidelijk koppelen; de huidige afzonderlijke undo-stromen mogen geen onverwachte stap terug opleveren.
- Modelleer downloaden, laden, segmenteren, genereren en opslaan als verschillende toestanden. Toon een echte voortgang waar die meetbaar is en anders de fase.
- Fouten noemen herstel: 'De beeldservice is gestopt. Opnieuw starten' in plaats van 'Failed to fetch'.
- Automatisch modelbeheer; details blijven zichtbaar voor diagnose. Geen herladen door iedere hover of elke kleine toolwisseling.
- Toon alternatieven pas als er een voorstel is. Recente resultaten horen in de bibliotheek, niet naast de huidige generatieparameters.
- Werkbaar bij kleinere vensters: de huidige minimum breedte van 780 px met hidden body-overflow verdient een responsive paneelstrategie.

## Techniek: behouden, aanscherpen en toevoegen

### Behouden

React/TypeScript/Vite, FastAPI/Pydantic, SQLite, Python modelproviders, CUDA/BF16/TF32, de seriele GPU-queue, native-resolutie compositor en documentrevisies. Tauri 2 blijft passend voor een installeerbare lokale app. Een nieuwe frontendframework- of databasekeuze lost de gevonden frictie niet op.

### Eerst aanscherpen

1. **Reproduceerbare runtime.** Pin de werkende Diffusers-commit of een geteste release, Torch/CUDA, Transformers en overige backenddependencies. Bewaar een lock/constraintsbestand en modelrevisies. Voer nieuwe modellen in een aparte evaluatieomgeving in; verander de bestaande omgeving niet om alleen 'latest' te halen.
2. **Capabilitycontract.** Modelleer generate, instruction-edit, native masked-fill, references, segment en outpaint afzonderlijk. Neem referentielimieten, alpha-output, ondersteunde parameters, licentie en gemeten hardwareprofiel op. Een compositorbeschermde edit is een andere capability dan native mask conditioning.
3. **Frontendownership.** Splits de grote App.tsx in editor shell, document/revisiestate, selectie/viewport en voorstel/jobstate. Een React reducer per duidelijke domeingrens is voldoende; een nieuwe state-library is geen voorwaarde.
4. **Jobevents.** Voeg een lokale SSE-stream toe voor echte fasen, tussenresultaten en terminale status. Behoud de bestaande cancel/supervisorroute. Dat maakt de UI duidelijker en voorkomt voortdurend grof pollen.
5. **Modelbeheer.** Behoud een zware generatiepipeline tegelijk. Cache SAM-embeddings per beeld/revisie en beheer de geheugenovergang naar generatie expliciet. Benchmark quantisatie en GPU-residentie voordat die default worden; BF16/SDPA blijft de bestaande baseline.

### Nieuwe productfuncties

1. **SAM 3-provider:** point/box/textselectie, add/subtract, en handmatige correctie. Meet kliklatency inclusief modelwisseling. Een hoverpreview moet uit gecachte informatie komen of gedebounced zijn.
2. **Betere regionale bewerking:** vergelijk huidige Klein plus compositor, native Fill en een mask-aware Klein/LanPaint-route. Test expliciete ruimtelijke conditionering; final compositing blijft de afdwingbare bescherming.
3. **Outpainting:** canvasafmetingen en bronplaatsing opslaan; lege ruimte en een interne context-/overlapmasker maken; resultaat als voorstelrevisie tonen. Onderhoud een aparte outputmasker zodat bestaande pixels bij 'behoud origineel' exact beschermd zijn. Grote uitbreidingen opdelen in gemeten stappen; hoeken en naden apart beoordelen.
4. **Referentiebeelden:** een tot de providerlimiet toegestane set met expliciete rol en verwijderbare thumbnails. Geef taakgebonden prompts aan de provider; haal geen kenmerken uit een willekeurige upload zonder dat die rol helder is.
5. **RGBA en afwerking:** werkruimte en compositor behouden alpha; lossless PNG-export voor transparantie. Achtergrondisolatie en edge-matting apart evalueren, omdat een selectiemasker niet automatisch perfecte haartransparantie oplevert. Voeg pas daarna een gemeten upscaleprovider toe.
6. **Desktoplevering:** Tauri beheert de lokale Python-worker, gezondheid, downloads, herstart en shutdown. Toon opstartproblemen in producttaal. De huidige shellscaffold is nog geen volledig geinstalleerd product.

ComfyUI is nuttig als evaluatiebank voor experimentele pipelines. Ik adviseer het nog niet als verplichte productiebackend: de bestaande providergrens kan de benodigde functies dragen zonder een extra proces en custom-node-dependencies. Alleen kiezen als een concrete ondersteunde workflow daarmee aantoonbaar eenvoudiger wordt.

## Volgorde en acceptatie

| Stap | Concrete uitkomst | Klaar wanneer |
|---|---|---|
| 1. UX en contract | Eén editor, begrijpelijke acties, heldere domeinstates en capabilitymodel | Nieuwe gebruiker opent/maakt een beeld, selecteert handmatig, beoordeelt een voorstel en exporteert zonder uitleg. |
| 2. Selectie en regionale kwaliteit | SAM-selectie plus gemeten masked-editroute | Selectie is corrigeerbaar; edits zijn taakgericht; beschermde pixels veranderen niet. |
| 3. Uitbreiden | Canvasranden en formaatpresets met outpaintvoorstellen | Alle randen en hoeken werken; na acceptatie, undo en heropenen kloppen canvas en bronplaatsing. |
| 4. Referenties en alpha | Onderwerp-/stijlreferenties, transparante documenten en achtergrondtools | De taak respecteert gekozen referenties; alpha overleeft bewerking en PNG-export. |
| 5. Leveren | Downloads, workerherstel en Windows-installer | Core flow werkt na een schone installatie zonder terminal. |

Pin eerst de huidige baseline zodat iedere stap eerlijk vergeleken kan worden. De installer is als laatste productmijlpaal vermeld, maar werk lifecycle en fouttoestanden vanaf de eerste stap mee uit.

### Kleine vaste evaluatieset

Gebruik publieke fixtures voor object verwijderen, object vervangen, haar/randen, materiaal wijzigen, product behouden, tekst wijzigen, achtergrond vervangen, referentie-identiteit, kleine maskers, en outpaint aan vier kanten plus een hoek. Neem kleine en 4K-bronbeelden op en herhaal representatieve edits meerdere keren om degradatie te zien.

Per route registreer je: cold-start en warme latency, segmentlatency, modelwisseltijd, eerste alternatief, totaal voor vier alternatieven, peak VRAM, RAM, outputafmetingen, verandering buiten het masker, naden, promptuitvoering en behoud van referentie. Leg modelrevisie, runtimeversies, resolutie en seeds vast.

Verifieer daarnaast: annuleren tijdens laden en genereren, herstel na OOM, export met alpha, revisieherstel na restart, en de eerste-gebruikerflow. Lokale snelheden en een betere modelrangorde blijven open totdat deze metingen bestaan.

## Reikwijdte van dit onderzoek

Dit voorstel volgt inspectie van de huidige repository en actuele primaire modelbronnen. Het begeleidende UI-concept simuleert selectie, voorstellen en canvasuitbreiding met een illustratief voorbeeldbeeld. Het is geen aangesloten AI-editor en bewijst geen modelkwaliteit. Runtimecode en de huidige modelomgeving zijn niet gewijzigd.

Verificatie: de bestaande frontendbuild en backendsyntaxcheck zijn geslaagd; lokale documentlinks zijn gecontroleerd. In het concept zijn voorstelkeuze, toepassen, undo, verwijderen, outpaintvoorbeeld, aantalselectie en de ontwerpwisseling gecontroleerd. Beide ontwerpen passen zonder horizontale overflow in een smalle weergave; Focus is ook gecontroleerd op 1024, 736, 360 en 320 px. Er zijn geen nieuwe modelkwaliteitstests of GPU-benchmarks uitgevoerd.
