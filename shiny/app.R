# =============================================================================
# OCBC Climate Risk — Shiny Interactive Dashboard
# Data: TerraClimate Indonesia (2021–2025)
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────

pkgs <- c(
  "shiny", "shinydashboard", "shinyWidgets",
  "tidyverse", "sf", "terra",
  "biscale", "cowplot",
  "forecast", "purrr",
  "scales", "DT"
)

for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  suppressPackageStartupMessages(library(p, character.only = TRUE))
}

set.seed(123)
options(scipen = 999)

# ── Paths (relative ke folder shiny/) ─────────────────────────────────────────

DATA_DIR  <- normalizePath("../data")
ADM1_SHP  <- file.path(DATA_DIR, "batas_provinsi", "Provinsi_Kemdagri.shp")
ADM2_SHP  <- file.path(DATA_DIR, "batas_kabupaten_kota",
                        "Batas Kabupaten Kemendagri 2024.shp")

# =============================================================================
# GLOBAL: Load & preprocess semua data sekali saat startup
# =============================================================================

message("=== Loading shapefiles ===")
indo_adm1 <- st_read(ADM1_SHP, quiet = TRUE) |> st_transform(4326)

# Buat ADM0 dari union ADM1
indo_adm0 <- st_union(indo_adm1) |> st_sf()
indo_vect  <- vect(indo_adm0)

message("=== Loading raster data (NetCDF) ===")

tmax_files <- list.files(DATA_DIR, pattern = "TerraClimate_tmax.*\\.nc$",
                         full.names = TRUE, recursive = FALSE)
tmin_files <- list.files(DATA_DIR, pattern = "TerraClimate_tmin.*\\.nc$",
                         full.names = TRUE, recursive = FALSE)
ppt_files  <- list.files(DATA_DIR, pattern = "TerraClimate_ppt.*\\.nc$",
                         full.names = TRUE, recursive = FALSE)

tmax_r <- rast(sort(tmax_files))
tmin_r <- rast(sort(tmin_files))
ppt_r  <- rast(sort(ppt_files))

# Crop & mask ke wilayah Indonesia
tmax_r <- crop(tmax_r, indo_vect) |> mask(indo_vect)
tmin_r <- crop(tmin_r, indo_vect) |> mask(indo_vect)
ppt_r  <- crop(ppt_r,  indo_vect) |> mask(indo_vect)

# Suhu rata-rata bulanan (tmax + tmin) / 2, skala 0.1 → °C
tavg_monthly <- (tmax_r + tmin_r) / 2 / 10
ppt_monthly  <- ppt_r

# Bangun time index
n_lyr     <- nlyr(tavg_monthly)
time_seq  <- seq(as.Date("2021-01-01"), by = "month", length.out = n_lyr)
years_chr <- format(time_seq, "%Y")
year_list <- sort(unique(years_chr))   # "2021" .. "2025"

message("=== Aggregating to annual ===")

tavg_annual <- tapp(tavg_monthly, index = years_chr, fun = mean, na.rm = TRUE)
ppt_annual  <- tapp(ppt_monthly,  index = years_chr, fun = sum,  na.rm = TRUE)
names(tavg_annual) <- paste0("tavg_", year_list)
names(ppt_annual)  <- paste0("ppt_",  year_list)

annual_mean_temp <- mean(tavg_annual, na.rm = TRUE)
annual_mean_ppt  <- mean(ppt_annual,  na.rm = TRUE)

# Stacked raster → data frame (digunakan K-Means & bivariat)
climate_stack <- c(annual_mean_temp, annual_mean_ppt)
names(climate_stack) <- c("suhu", "curah_hujan")
climate_df <- as.data.frame(climate_stack, xy = TRUE, na.rm = TRUE)

# ── Per-tahun data frames untuk peta dinamis ──────────────────────────────────

message("=== Converting rasters to per-year data frames ===")

tavg_df_list <- setNames(lapply(year_list, function(yr) {
  df <- as.data.frame(tavg_annual[[paste0("tavg_", yr)]], xy = TRUE, na.rm = TRUE)
  names(df)[3] <- "value"
  df
}), year_list)

ppt_df_list <- setNames(lapply(year_list, function(yr) {
  df <- as.data.frame(ppt_annual[[paste0("ppt_", yr)]], xy = TRUE, na.rm = TRUE)
  names(df)[3] <- "value"
  df
}), year_list)

# ── Annual summary (untuk sparkline tren) ─────────────────────────────────────

temp_annual_df <- data.frame(
  year      = as.integer(year_list),
  temp_mean = as.numeric(global(tavg_annual, fun = mean, na.rm = TRUE)$mean)
)

ppt_annual_df <- data.frame(
  year      = as.integer(year_list),
  precip_mm = as.numeric(global(ppt_annual,  fun = mean, na.rm = TRUE)$mean)
)

# ── Bivariat ──────────────────────────────────────────────────────────────────

climate_bi <- bi_class(climate_df, x = suhu, y = curah_hujan,
                        style = "quantile", dim = 4)

# ── K-Means: pre-compute elbow WSS (mahal, cukup sekali) ─────────────────────

message("=== Pre-computing K-Means elbow ===")

features_mat <- scale(climate_df[, c("suhu", "curah_hujan")])

wss_df <- data.frame(
  k   = 2:10,
  wss = map_dbl(2:10, function(k) {
    set.seed(123)
    kmeans(features_mat, centers = k, nstart = 25, iter.max = 100)$tot.withinss
  })
)

# ── ARIMA: fit model sekali (mahal) ───────────────────────────────────────────

message("=== Fitting ARIMA models ===")

temp_ts_m <- ts(
  as.numeric(global(tavg_monthly, fun = mean, na.rm = TRUE)$mean),
  start = c(2021, 1), frequency = 12
)
ppt_ts_m <- ts(
  as.numeric(global(ppt_monthly, fun = mean, na.rm = TRUE)$mean),
  start = c(2021, 1), frequency = 12
)

temp_arima_fit <- auto.arima(temp_ts_m, seasonal = TRUE,
                              stepwise = FALSE, approximation = FALSE)
ppt_arima_fit  <- auto.arima(ppt_ts_m,  seasonal = TRUE,
                              stepwise = FALSE, approximation = FALSE)

message("=== Data ready. Launching app ===")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

theme_map <- function() {
  theme_void(base_size = 13) +
    theme(
      plot.title    = element_text(hjust = 0.5, face = "bold", size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 11, color = "gray40"),
      plot.caption  = element_text(size = 9, color = "gray50"),
      legend.position = "right"
    )
}

overlay_adm <- function(p) {
  p +
    geom_sf(data = indo_adm1, fill = NA, color = "black", linewidth = 0.18) +
    geom_sf(data = indo_adm0, fill = NA, color = "black", linewidth = 0.45)
}

# =============================================================================
# UI
# =============================================================================

ui <- dashboardPage(
  skin = "blue",

  # ── Header ──────────────────────────────────────────────────────────────────
  dashboardHeader(
    title = tags$span(
      icon("cloud-sun"), " OCBC Climate Risk"
    ),
    titleWidth = 260
  ),

  # ── Sidebar ─────────────────────────────────────────────────────────────────
  dashboardSidebar(
    width = 230,
    sidebarMenu(
      id = "sidebar",
      menuItem("Beranda",           tabName = "home",    icon = icon("home")),
      menuItem("Peta Iklim Tahunan",tabName = "peta",    icon = icon("map")),
      menuItem("Peta Bivariat",     tabName = "bivariat",icon = icon("layer-group")),
      menuItem("K-Means Clustering",tabName = "kmeans",  icon = icon("object-group")),
      menuItem("ARIMA Forecast",    tabName = "arima",   icon = icon("chart-line"))
    ),
    tags$hr(),
    tags$div(
      style = "padding: 10px 15px; font-size: 11px; color: #aaa;",
      "Data: TerraClimate 2021–2025", tags$br(),
      "Cakupan: Indonesia (ADM1)"
    )
  ),

  # ── Body ────────────────────────────────────────────────────────────────────
  dashboardBody(
    tags$head(tags$style(HTML("
      .content-wrapper { background-color: #f4f6f9; }
      .box             { border-radius: 6px; }
      .info-box        { border-radius: 6px; }
      .small-box       { border-radius: 6px; }
      .nav-tabs-custom > .tab-content { padding: 10px 0 0 0; }
    "))),

    tabItems(

      # ══════════════════════════════════════════════════════════════════════
      # TAB: BERANDA
      # ══════════════════════════════════════════════════════════════════════
      tabItem(tabName = "home",
        fluidRow(
          column(12,
            tags$h2("AI-Guided Climate Risk Adaptation",
                    style = "font-weight:bold; margin-bottom:4px"),
            tags$p("Dashboard interaktif analisis risiko iklim Indonesia — TerraClimate (2021–2025)",
                   style = "color:gray; margin-bottom:16px")
          )
        ),
        fluidRow(
          valueBoxOutput("vbox_tahun",   width = 3),
          valueBoxOutput("vbox_suhu",    width = 3),
          valueBoxOutput("vbox_hujan",   width = 3),
          valueBoxOutput("vbox_provinsi",width = 3)
        ),
        fluidRow(
          box(
            title = "Tren Suhu Rata-rata Tahunan (°C)",
            status = "danger", solidHeader = TRUE, width = 6,
            plotOutput("home_temp", height = "250px")
          ),
          box(
            title = "Tren Curah Hujan Rata-rata Tahunan (mm)",
            status = "info", solidHeader = TRUE, width = 6,
            plotOutput("home_ppt", height = "250px")
          )
        )
      ),

      # ══════════════════════════════════════════════════════════════════════
      # TAB: PETA IKLIM TAHUNAN
      # ══════════════════════════════════════════════════════════════════════
      tabItem(tabName = "peta",
        fluidRow(
          # Panel kontrol
          box(
            title = "Kontrol", status = "primary", solidHeader = TRUE, width = 3,
            radioButtons("peta_var", "Variabel:",
              choices  = c("Suhu (°C)" = "suhu", "Curah Hujan (mm)" = "ppt"),
              selected = "suhu"
            ),
            tags$hr(),
            sliderInput("peta_tahun", "Tahun:",
              min   = as.integer(min(year_list)),
              max   = as.integer(max(year_list)),
              value = as.integer(min(year_list)),
              step  = 1, sep = "",
              animate = animationOptions(interval = 1800, loop = TRUE)
            ),
            tags$p("Klik ▶ untuk animasi otomatis.",
                   style = "font-size:11px; color:gray; margin-top:4px"),
            tags$hr(),
            tags$h5("Statistik Nasional"),
            tableOutput("peta_stats")
          ),
          # Peta
          box(
            title = uiOutput("peta_map_title"),
            status = "primary", solidHeader = TRUE, width = 9,
            plotOutput("peta_map", height = "480px")
          )
        ),
        fluidRow(
          box(
            title = "Tren Rata-rata Nasional (garis vertikal = tahun dipilih)",
            status = "info", solidHeader = TRUE, width = 12,
            plotOutput("peta_trend", height = "180px")
          )
        )
      ),

      # ══════════════════════════════════════════════════════════════════════
      # TAB: PETA BIVARIAT
      # ══════════════════════════════════════════════════════════════════════
      tabItem(tabName = "bivariat",
        fluidRow(
          box(
            title = "Peta Bivariat: Suhu × Curah Hujan (Rata-rata 2021–2025)",
            status = "warning", solidHeader = TRUE, width = 12,
            plotOutput("biv_map", height = "620px")
          )
        ),
        fluidRow(
          box(
            title = "Distribusi Kelas Bivariat",
            status = "warning", solidHeader = TRUE, width = 6,
            plotOutput("biv_dist", height = "250px")
          ),
          box(
            title = "Interpretasi Skema Warna",
            status = "warning", solidHeader = TRUE, width = 6,
            tags$p("Skema", tags$b("BlueOr"), "(Biru–Oranye) 4×4 kelas:"),
            tags$ul(
              tags$li(tags$b("Biru gelap:"), " Suhu rendah + curah hujan tinggi"),
              tags$li(tags$b("Oranye gelap:"), " Suhu tinggi + curah hujan rendah"),
              tags$li(tags$b("Coklat:"), " Suhu tinggi + curah hujan tinggi"),
              tags$li(tags$b("Biru muda:"), " Suhu rendah + curah hujan rendah")
            ),
            tags$p("16 kelas total = 4 kelas suhu × 4 kelas curah hujan (quantile).",
                   style = "color:gray; font-size:12px")
          )
        )
      ),

      # ══════════════════════════════════════════════════════════════════════
      # TAB: K-MEANS
      # ══════════════════════════════════════════════════════════════════════
      tabItem(tabName = "kmeans",
        fluidRow(
          # Panel kontrol + elbow
          box(
            title = "Kontrol K-Means", status = "success",
            solidHeader = TRUE, width = 3,
            sliderInput("km_k", "Jumlah Cluster (k):",
                        min = 2, max = 10, value = 5, step = 1),
            tags$p("Pilih k di 'siku' kurva di bawah.",
                   style = "font-size:11px; color:gray"),
            tags$hr(),
            tags$h5("Elbow Curve"),
            plotOutput("km_elbow", height = "200px"),
            tags$hr(),
            tags$h5("Ringkasan Cluster"),
            DT::dataTableOutput("km_table")
          ),
          # Peta cluster
          box(
            title = uiOutput("km_map_title"),
            status = "success", solidHeader = TRUE, width = 9,
            plotOutput("km_map", height = "500px")
          )
        ),
        fluidRow(
          box(
            title = "Profil Cluster: Suhu vs Curah Hujan (sampel 5.000 piksel)",
            status = "success", solidHeader = TRUE, width = 12,
            plotOutput("km_scatter", height = "320px")
          )
        )
      ),

      # ══════════════════════════════════════════════════════════════════════
      # TAB: ARIMA
      # ══════════════════════════════════════════════════════════════════════
      tabItem(tabName = "arima",
        fluidRow(
          # Kontrol
          box(
            title = "Kontrol ARIMA", status = "danger",
            solidHeader = TRUE, width = 3,
            radioButtons("arima_var", "Variabel:",
              choices  = c("Suhu (°C)" = "suhu", "Curah Hujan (mm)" = "ppt"),
              selected = "suhu"
            ),
            tags$hr(),
            sliderInput("arima_h", "Horizon Forecast (bulan):",
                        min = 12, max = 60, value = 36, step = 12,
                        post = " bln"),
            tags$hr(),
            tags$h5("Spesifikasi Model"),
            verbatimTextOutput("arima_spec"),
            tags$hr(),
            tags$h5("Akurasi In-Sample"),
            tableOutput("arima_accuracy")
          ),
          # Forecast plot
          box(
            title = uiOutput("arima_fc_title"),
            status = "danger", solidHeader = TRUE, width = 9,
            plotOutput("arima_fc_plot", height = "420px")
          )
        ),
        fluidRow(
          box(
            title = "Diagnostik Residual",
            status = "warning", solidHeader = TRUE, width = 12,
            plotOutput("arima_resid", height = "300px")
          )
        )
      )

    ) # end tabItems
  )   # end dashboardBody
)     # end dashboardPage

# =============================================================================
# SERVER
# =============================================================================

server <- function(input, output, session) {

  # ── VALUE BOXES (Beranda) ────────────────────────────────────────────────────

  output$vbox_tahun <- renderValueBox({
    valueBox("2021–2025", "Periode Data (Bulanan)", icon = icon("calendar"),
             color = "blue")
  })
  output$vbox_suhu <- renderValueBox({
    v <- round(mean(temp_annual_df$temp_mean), 1)
    valueBox(paste0(v, " °C"), "Rata-rata Suhu Nasional", icon = icon("thermometer-half"),
             color = "red")
  })
  output$vbox_hujan <- renderValueBox({
    v <- round(mean(ppt_annual_df$precip_mm), 0)
    valueBox(paste0(format(v, big.mark = "."), " mm"), "Rata-rata Curah Hujan Nasional",
             icon = icon("cloud-rain"), color = "aqua")
  })
  output$vbox_provinsi <- renderValueBox({
    valueBox(nrow(indo_adm1), "Provinsi Tercakup", icon = icon("map-marker"),
             color = "green")
  })

  # ── BERANDA TRENDS ───────────────────────────────────────────────────────────

  output$home_temp <- renderPlot({
    ggplot(temp_annual_df, aes(x = year, y = temp_mean)) +
      geom_line(color = "#e74c3c", linewidth = 1.2) +
      geom_point(color = "#e74c3c", size = 3.5) +
      geom_smooth(method = "lm", se = TRUE, color = "#c0392b",
                  fill = "#fadbd8", linetype = "dashed", linewidth = 0.8) +
      scale_x_continuous(breaks = as.integer(year_list)) +
      labs(x = "Tahun", y = "Suhu (°C)") +
      theme_minimal(base_size = 12)
  })

  output$home_ppt <- renderPlot({
    ggplot(ppt_annual_df, aes(x = year, y = precip_mm)) +
      geom_line(color = "#2980b9", linewidth = 1.2) +
      geom_point(color = "#2980b9", size = 3.5) +
      geom_smooth(method = "lm", se = TRUE, color = "#1a5276",
                  fill = "#d6eaf8", linetype = "dashed", linewidth = 0.8) +
      scale_x_continuous(breaks = as.integer(year_list)) +
      labs(x = "Tahun", y = "Curah Hujan (mm)") +
      theme_minimal(base_size = 12)
  })

  # ── PETA IKLIM TAHUNAN ───────────────────────────────────────────────────────

  sel_df <- reactive({
    yr <- as.character(input$peta_tahun)
    if (input$peta_var == "suhu") tavg_df_list[[yr]] else ppt_df_list[[yr]]
  })

  output$peta_map_title <- renderUI({
    lbl <- if (input$peta_var == "suhu") "Suhu Rata-rata (°C)" else "Total Curah Hujan (mm)"
    paste(lbl, "—", input$peta_tahun)
  })

  output$peta_map <- renderPlot({
    df  <- sel_df()
    yr  <- input$peta_tahun
    var <- input$peta_var

    if (var == "suhu") {
      pal   <- scale_fill_distiller(palette = "RdYlBu", direction = -1,
                                    name = "Suhu (°C)")
      sub   <- paste("Rata-rata suhu bulanan tahun", yr)
    } else {
      pal   <- scale_fill_distiller(palette = "Blues", direction = 1,
                                    name = "Curah\nHujan (mm)")
      sub   <- paste("Total curah hujan tahunan", yr)
    }

    p <- ggplot() +
      geom_raster(data = df, aes(x = x, y = y, fill = value)) +
      pal
    p <- overlay_adm(p)
    p + labs(subtitle = sub, caption = "Sumber: TerraClimate") +
      theme_map()
  })

  output$peta_trend <- renderPlot({
    if (input$peta_var == "suhu") {
      df   <- temp_annual_df; yvar <- "temp_mean"
      col  <- "#e74c3c"; ylab <- "Suhu (°C)"
    } else {
      df   <- ppt_annual_df; yvar <- "precip_mm"
      col  <- "#2980b9"; ylab <- "Curah Hujan (mm)"
    }
    yr_sel <- input$peta_tahun

    ggplot(df, aes(x = year, y = .data[[yvar]])) +
      geom_line(color = col, linewidth = 1) +
      geom_point(color = col, size = 2.5) +
      geom_vline(xintercept = yr_sel, linetype = "dashed", color = "gray40", linewidth = 0.8) +
      geom_point(
        data = df[df$year == yr_sel, ],
        aes(x = year, y = .data[[yvar]]),
        shape = 21, size = 5, color = "black", fill = col
      ) +
      scale_x_continuous(breaks = as.integer(year_list)) +
      labs(x = "Tahun", y = ylab) +
      theme_minimal(base_size = 11)
  })

  output$peta_stats <- renderTable({
    df <- sel_df()
    data.frame(
      Statistik = c("Min", "Mean", "Max"),
      Nilai     = round(c(min(df$value, na.rm=TRUE),
                          mean(df$value, na.rm=TRUE),
                          max(df$value, na.rm=TRUE)), 2)
    )
  }, striped = TRUE, hover = TRUE, bordered = TRUE, spacing = "xs")

  # ── BIVARIAT ─────────────────────────────────────────────────────────────────

  output$biv_map <- renderPlot({
    pal <- "BlueOr"

    map_biv <- ggplot() +
      geom_raster(data = climate_bi, aes(x = x, y = y, fill = bi_class),
                  show.legend = FALSE)

    map_biv <- overlay_adm(map_biv)

    map_biv <- map_biv +
      bi_scale_fill(pal = pal, dim = 4) +
      labs(
        title    = "Indonesia: Pola Suhu dan Curah Hujan",
        subtitle = "Rata-rata 2021–2025 | Klasifikasi Bivariat Quantile (4×4)",
        caption  = "Sumber: TerraClimate | Author: Alfa Pradana"
      ) +
      theme_map()

    legend_biv <- bi_legend(
      pal  = pal, dim = 4,
      xlab = "Suhu (°C)", ylab = "Curah Hujan (mm)", size = 9
    )

    cowplot::ggdraw() +
      cowplot::draw_plot(map_biv, 0, 0, 1, 1) +
      cowplot::draw_plot(legend_biv, 0.01, 0.05, 0.20, 0.20)
  })

  output$biv_dist <- renderPlot({
    climate_bi |>
      count(bi_class) |>
      ggplot(aes(x = reorder(bi_class, -n), y = n)) +
      geom_col(fill = "#3498db", alpha = 0.8) +
      labs(x = "Kelas Bivariat", y = "Jumlah Piksel") +
      theme_minimal(base_size = 11) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))
  })

  # ── K-MEANS ──────────────────────────────────────────────────────────────────

  km_result <- reactive({
    set.seed(123)
    kmeans(features_mat, centers = input$km_k, nstart = 25, iter.max = 100)
  })

  output$km_elbow <- renderPlot({
    ggplot(wss_df, aes(x = k, y = wss)) +
      geom_line(linewidth = 0.8) +
      geom_point(size = 2.5) +
      geom_point(
        data = wss_df[wss_df$k == input$km_k, ],
        aes(x = k, y = wss), color = "#27ae60", size = 5
      ) +
      scale_x_continuous(breaks = 2:10) +
      labs(x = "k", y = "WSS") +
      theme_minimal(base_size = 10)
  })

  output$km_map_title <- renderUI({
    paste("Peta Zona Iklim — K-Means (k =", input$km_k, ")")
  })

  output$km_map <- renderPlot({
    km <- km_result()
    df <- climate_df
    df$cluster <- as.factor(km$cluster)

    p <- ggplot() +
      geom_raster(data = df, aes(x = x, y = y, fill = cluster))
    p <- overlay_adm(p)
    p +
      scale_fill_brewer(palette = "Set2", name = "Zona Iklim") +
      labs(
        subtitle = "Berdasarkan suhu dan curah hujan rata-rata 2021–2025",
        caption  = "Sumber: TerraClimate"
      ) +
      theme_map() +
      theme(legend.position = "bottom",
            legend.title     = element_text(size = 11))
  })

  output$km_scatter <- renderPlot({
    km  <- km_result()
    idx <- sample(nrow(climate_df), min(8000, nrow(climate_df)))
    sdf <- climate_df[idx, ]
    sdf$cluster <- as.factor(km$cluster[idx])

    ggplot(sdf, aes(x = suhu, y = curah_hujan, color = cluster)) +
      geom_point(alpha = 0.45, size = 1.2) +
      scale_color_brewer(palette = "Set2", name = "Cluster") +
      stat_ellipse(level = 0.85, linewidth = 0.7) +
      labs(x = "Suhu Rata-rata (°C)", y = "Curah Hujan Rata-rata (mm/tahun)") +
      theme_minimal(base_size = 12) +
      theme(legend.position = "right")
  })

  output$km_table <- DT::renderDataTable({
    km <- km_result()
    df <- climate_df
    df$cluster <- km$cluster

    df |>
      group_by(Cluster = cluster) |>
      summarise(
        `Piksel`       = n(),
        `Suhu (°C)`    = round(mean(suhu), 2),
        `Hujan (mm)`   = round(mean(curah_hujan), 0),
        .groups = "drop"
      ) |>
      arrange(Cluster) |>
      DT::datatable(
        options  = list(pageLength = 12, dom = "t", ordering = FALSE),
        rownames = FALSE
      )
  })

  # ── ARIMA ────────────────────────────────────────────────────────────────────

  arima_model <- reactive({
    if (input$arima_var == "suhu") temp_arima_fit else ppt_arima_fit
  })

  arima_fc <- reactive({
    forecast(arima_model(), h = input$arima_h)
  })

  output$arima_fc_title <- renderUI({
    var_lbl <- if (input$arima_var == "suhu") "Suhu (°C)" else "Curah Hujan (mm)"
    paste("Forecast", var_lbl, "—", input$arima_h, "bulan ke depan")
  })

  output$arima_spec <- renderText({
    m <- arima_model()
    paste0(
      "Metode : ", m$method, "\n",
      "AIC    : ", round(AIC(m), 2),   "\n",
      "BIC    : ", round(BIC(m), 2)
    )
  })

  output$arima_accuracy <- renderTable({
    acc <- as.data.frame(accuracy(arima_model()))
    data.frame(
      Metrik = c("RMSE", "MAE", "MAPE"),
      Nilai  = round(c(acc$RMSE, acc$MAE, acc$MAPE), 4)
    )
  }, striped = TRUE, hover = TRUE, bordered = TRUE, spacing = "xs")

  output$arima_fc_plot <- renderPlot({
    fc   <- arima_fc()
    var  <- input$arima_var
    ylab <- if (var == "suhu") "Suhu (°C)" else "Curah Hujan (mm)"
    col  <- if (var == "suhu") "#e74c3c" else "#2980b9"

    autoplot(fc, include = n_lyr) +
      labs(x = "Tahun", y = ylab,
           caption = "Shaded area: interval kepercayaan 80% (gelap) dan 95% (terang)") +
      scale_color_manual(values = col) +
      theme_minimal(base_size = 13) +
      theme(plot.title    = element_blank(),
            legend.position = "none")
  })

  output$arima_resid <- renderPlot({
    checkresiduals(arima_model(), plot = TRUE, theme = theme_minimal())
  })

}

# =============================================================================
shinyApp(ui, server)
