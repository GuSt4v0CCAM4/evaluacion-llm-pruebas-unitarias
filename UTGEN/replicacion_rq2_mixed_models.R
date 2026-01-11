# Replicación RQ2: Análisis Estadístico Completo con Mixed Models
# Cumulative Link Mixed Models para Fixed Bugs
# Generalized Linear Mixed Models (Gamma) para Time

# Instalar paquetes si es necesario
# install.packages(c("ordinal", "lme4", "car", "emmeans"))

library(ordinal)
library(lme4)
library(car)
library(emmeans)

# Cargar datos
results <- read.csv("UTGen/Results/RQ2/results.csv", stringsAsFactors = FALSE)

# Preparar datos
results$Technique <- factor(results$TestGen, levels = c("Evo", "UTG"))
results$Object <- factor(results$Task)
results$Period <- factor(results$Question)
results$Order <- factor(ifelse(results$`# Variety` %% 2 == 1, 1, 2))
results$ParticipantID <- factor(results$`#`)

# Convertir tiempo a minutos
parse_time <- function(time_str) {
  parts <- strsplit(time_str, ":")[[1]]
  if (length(parts) == 3) {
    hours <- as.numeric(parts[1])
    minutes <- as.numeric(parts[2])
    seconds <- as.numeric(parts[3])
    return(hours * 60 + minutes + seconds / 60)
  } else if (length(parts) == 2) {
    minutes <- as.numeric(parts[1])
    seconds <- as.numeric(parts[2])
    return(minutes + seconds / 60)
  }
  return(NA)
}

results$time_minutes <- sapply(results$time, parse_time)

# Factor para bugs arreglados (ordinal)
results$bugs_fixed <- factor(results$`# bugs fixed`, levels = 0:4, ordered = TRUE)

cat("=== RQ2: Análisis con Mixed Models ===\n\n")

# Modelo 1: Fixed Bugs (Cumulative Link Mixed Model)
cat("Modelo 1: Fixed Bugs (Cumulative Link Mixed Model)\n")
cat("---------------------------------------------------\n")

model_bugs <- clmm(bugs_fixed ~ Technique + Object + Technique:Object + 
                   Order + Period + (1|ParticipantID),
                   data = results)

cat("\nResumen del modelo:\n")
print(summary(model_bugs))

cat("\nAnova del modelo:\n")
print(Anova(model_bugs, type = "II"))

# Post-hoc analysis para Technique
cat("\nPost-hoc: Comparación por Technique\n")
emm_bugs <- emmeans(model_bugs, ~ Technique)
print(pairs(emm_bugs))

# Modelo 2: Time (Gamma GLMM)
cat("\n\nModelo 2: Time (Gamma GLMM)\n")
cat("---------------------------------------------------\n")

# Filtrar valores válidos de tiempo
results_time <- results[!is.na(results$time_minutes) & results$time_minutes > 0, ]

model_time <- glmer(time_minutes ~ Technique + Object + Technique:Object + 
                    Order + Period + (1|ParticipantID),
                    family = Gamma(link = "log"),
                    data = results_time,
                    control = glmerControl(optimizer = "bobyqa"))

cat("\nResumen del modelo:\n")
print(summary(model_time))

cat("\nAnova del modelo:\n")
print(Anova(model_time, type = "II"))

# Post-hoc analysis para Technique
cat("\nPost-hoc: Comparación por Technique\n")
emm_time <- emmeans(model_time, ~ Technique, type = "response")
print(pairs(emm_time))

# Guardar resultados
cat("\n\n=== Guardando resultados ===\n")
sink("resultados_replicacion/rq2_mixed_models_output.txt")
cat("=== RQ2: Análisis con Mixed Models ===\n\n")
cat("Modelo 1: Fixed Bugs\n")
print(summary(model_bugs))
print(Anova(model_bugs, type = "II"))
cat("\nModelo 2: Time\n")
print(summary(model_time))
print(Anova(model_time, type = "II"))
sink()

cat("Resultados guardados en: resultados_replicacion/rq2_mixed_models_output.txt\n")

