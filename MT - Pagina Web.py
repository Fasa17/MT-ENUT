# ----------
# Librerías necesarias para la aplicación
# ----------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------
# Configuración global de la aplicación Streamlit
# ----------
st.set_page_config(
    page_title = "Encuesta Nacional sobre Uso del Tiempo",
    page_icon = "https://i.imgur.com/2ZnBdH3.png",
    layout = "wide",
    initial_sidebar_state="expanded"
)

# ----------
# Funciones
# ----------

# ----------
# Función para cargar las bases de datos desde Google Drive
# ----------
@st.cache_data
def Cargando_Datos():
    try:
        url_2015 = "https://drive.google.com/uc?export=download&id=1jjtdmGdcGkZV01SBb7RRsi_Ct9pf-YxS"
        url_2023 = "https://drive.google.com/uc?export=download&id=1qtzT6C5H26DWtVXHMMzBBx8ugtgehrKc"
        df_15 = pd.read_csv(url_2015)
        df_23 = pd.read_csv(url_2023)
        return df_15, df_23
    except Exception as e:
        st.error("Error al cargar la base de datos. Por favor, revise su conexión o intente más tarde.")
        return pd.DataFrame(), pd.DataFrame()
    
# ----------
# Calcular día tipo ponderando semana y fin de semana
# ----------
def dia_tipo(df, variables):
    df = df.copy()
    for v in variables:
        col_ds = f"{v}_ds"
        col_fds = f"{v}_fds"
        col_dt = f"{v}_dt"
        if col_ds in df.columns and col_fds in df.columns:
            df[col_dt] = df[col_ds].fillna(0) * 5/7 + df[col_fds].fillna(0) * 2/7
    return df

# ----------
# Convertir valor decimal en formato hora (HH:MM)
# ----------
def dec_a_hhmm(decimal_hora):
    if pd.isna(decimal_hora):
        return "00:00"
    horas = int(decimal_hora)
    minutos = int(round((decimal_hora - horas) * 60))
    if minutos == 60:
        horas += 1
        minutos = 0
    return f"{horas:02d}:{minutos:02d}"
    
# ----------
# Función para crear gráfico de barras de porcentaje por sexo
# ----------
def grafico_barras(df, variables, titulo):
    color_hombres = "#5B9BD5"
    color_mujeres = "#ED97C0"
    sexo_map = {1: "Hombres", 2: "Mujeres"}
    df_filtrado = df[df["edad"] > 15].copy()
    condiciones = np.logical_or.reduce([df_filtrado[var] == 1 for var in variables])
    df_filtrado.loc[:, "actividad_ds_fds"] = 0
    df_filtrado.loc[condiciones, "actividad_ds_fds"] = 1
    df_perc = df_filtrado.groupby("sexo")["actividad_ds_fds"].mean().reset_index()
    df_perc["Porcentaje"] = df_perc["actividad_ds_fds"] * 100
    df_perc["sexo"] = df_perc["sexo"].map(sexo_map)
    fig = px.bar(
        df_perc,
        x="sexo",
        y="Porcentaje",
        text="Porcentaje",
        color="sexo",
        color_discrete_map={"Hombres": color_hombres, "Mujeres": color_mujeres},
        labels={"sexo": "Sexo", "Porcentaje": "Porcentaje (%)"},
        title=titulo,
    )
    fig.update_traces(
        texttemplate='%{text:.0f}%',
        textposition='outside',
        hovertemplate="<b>Sexo:</b> %{x}<br><b>Porcentaje:</b> %{y:.2f}%<extra></extra>",
        showlegend=False
    )
    fig.update_layout(
        yaxis=dict(range=[0, 110]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        <div style='text-align: right; margin-top: -40px;'>
            <img src='https://i.imgur.com/wiPCMeE.png' width='50' style='border-radius: 5px;'>
        </div>
        """,
        unsafe_allow_html=True
    )
    
# ----------
# Función para graficar área apilada de tiempo dedicado a actividades por edad
# ----------
def graficar_area_apilada(df_15, df_23, variables, nomb_act, regiones):
    cols = st.columns(4)
    with cols[0]:
        anio = st.selectbox("Seleccione el año de interés:", ["2015", "2023"], index=1, key="anio_area")
    with cols[1]:
        sexo = st.selectbox("Seleccione sexo:", ["Total", "Hombres", "Mujeres"], index=0, key="sexo_area")
    with cols[2]:
        tipo_dia = st.selectbox("Seleccione tipo de día:", ("Día tipo", "Día de semana", "Día de fin de semana"), index=0, key="tipodia_area")
    with cols[3]:
        region = st.selectbox("Seleccione región:", regiones, key="region_area")
    sufijo_dia = {
        "Día de semana": "_ds",
        "Día de fin de semana": "_fds",
        "Día tipo": "_dt"
    }[tipo_dia]
    df_base = df_15.copy() if anio == "2015" else df_23.copy()
    if region.lower() != "todas":
        df_base = df_base[df_base["id_reg"].str.lower() == region.lower()]
    if anio == "2015" and region.lower() == "ñuble":
        st.warning("⚠️ No existen datos disponibles para la región de Ñuble en el año 2015.")
        return
    if sexo == "Hombres":
        df_base = df_base[df_base["sexo"] == 1]
    elif sexo == "Mujeres":
        df_base = df_base[df_base["sexo"] == 2]
    df_processed = dia_tipo(df_base, variables)
    col_utdd = [f"{v}{sufijo_dia}" for v in variables]
    df_processed[col_utdd] = df_processed[col_utdd].fillna(0)
    edad_col = "edad"
    df_processed[edad_col] = pd.to_numeric(df_processed[edad_col], errors='coerce').clip(upper=80)
    df_prom_act_edad = df_processed.groupby(edad_col)[col_utdd].mean().reset_index()
    df_long_act_edad = pd.melt(
        df_prom_act_edad,
        id_vars=edad_col,
        value_vars=col_utdd,
        var_name="Actividad",
        value_name="Tiempo"
    )
    df_long_act_edad["Actividad"] = df_long_act_edad["Actividad"].str.replace(sufijo_dia, "", regex=False)
    df_long_act_edad["Actividad"] = df_long_act_edad["Actividad"].map(nomb_act)
    orden_act = [nomb_act[v] for v in variables if v in nomb_act]
    df_long_act_edad["Actividad"] = pd.Categorical(df_long_act_edad["Actividad"], categories=orden_act, ordered=True)
    df_long_act_edad = df_long_act_edad.sort_values([edad_col, "Actividad"])
    df_long_act_edad["Tiempo"] = pd.to_numeric(df_long_act_edad["Tiempo"], errors="coerce").fillna(0)
    df_long_act_edad["Tiempohhmm"] = df_long_act_edad["Tiempo"].apply(dec_a_hhmm)
    fig_act_dia = px.area(
        df_long_act_edad,
        x=edad_col,
        y="Tiempo",
        color="Actividad",
        labels={edad_col: "Edad", "Tiempo": "Horas"},
        custom_data=["Tiempohhmm", edad_col, "Actividad"],
        hover_data={"Tiempo": False, "Tiempohhmm": True, edad_col: True, "Actividad": True},
    )
    fig_act_dia.update_traces(
        hovertemplate=(
            "<b>Edad:</b> %{customdata[1]} años<br>"
            "<b>Actividad:</b> %{customdata[2]}<br>"
            "<b>Horas:</b> %{customdata[0]}<extra></extra>"
        )
    )
    fig_act_dia.update_layout(
        xaxis=dict(range=[12, df_long_act_edad[edad_col].max()]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=700,
        height=500,
        legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_act_dia, use_container_width=True)
    st.markdown(
        """
        <div style='text-align: right; margin-top: -15px;'>
            <img src='https://i.imgur.com/wiPCMeE.png' width='50' style='border-radius: 5px;'>
        </div>
        """,
        unsafe_allow_html=True
    )
    
# ----------
# Función para comparar dos actividades seleccionadas por usuario y graficar su evolución por edad
# ----------
def comparar_actividades(df_15, df_23, variables, nomb_act, regiones):
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Besley&display=swap');
        .titulo-secundario {
            font-family: 'Besley', serif;
            font-size: 1.4em;
            color: black;
            font-weight: 600;
        }
        .subtitulo-secundario {
            font-family: Arial, sans-serif;
            font-size: 1.0em;
            color: black;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    actividades_permitidas = {k: v for k, v in nomb_act.items() if k in variables}
    nomb_act_inv = {v: k for k, v in actividades_permitidas.items()}
    def controles_linea(n):
        st.markdown(f'<div class="subtitulo-secundario">Parámetros actividad Nº {n}</div>', unsafe_allow_html=True)
        cols1 = st.columns(2)
        with cols1[0]:
            act = st.selectbox(
                f"Seleccione actividad de interés:", list(actividades_permitidas.values()),
                key=f"act_{n}"
            )
        with cols1[1]:
            anio_default_index = 0 if n == 1 else 1
            anio = st.selectbox(
                f"Seleccione el año de interés:", ["2015", "2023"],
                key=f"anio_{n}",
                index=anio_default_index
            )
        cols2 = st.columns(3)
        with cols2[0]:
            sexo = st.selectbox(
                f"Seleccione sexo:", ["Total", "Hombres", "Mujeres"],
                key=f"sexo_{n}"
            )
        with cols2[1]:
            tipo_dia = st.selectbox(
                f"Seleccione tipo de día:",
                ["Día tipo", "Día de semana", "Día de fin de semana"],
                key=f"dia_{n}"
            )
        with cols2[2]:
            region = st.selectbox(
                f"Seleccione región:",
                regiones,
                key=f"region_{n}"
            )
        return {
            "actividad": act,
            "anio": anio,
            "sexo": sexo,
            "tipo_dia": tipo_dia,
            "region": region
        }
    filtros_1 = controles_linea(1)
    filtros_2 = controles_linea(2)
    sufijos = {
        "Día de semana": "_ds",
        "Día de fin de semana": "_fds",
        "Día tipo": "_dt"
    }
    def preparar_df(df, filtros, etiqueta_linea):
        cod = nomb_act_inv[filtros["actividad"]]
        col = f"{cod}{sufijos[filtros['tipo_dia']]}"        
        df_temp = df.copy()
        if filtros["region"] != "Todas":
            df_temp = df_temp[df_temp["id_reg"] == filtros["region"]]
        if filtros["sexo"] == "Hombres":
            df_temp = df_temp[df_temp["sexo"] == 1]
        elif filtros["sexo"] == "Mujeres":
            df_temp = df_temp[df_temp["sexo"] == 2]
        df_temp["edad"] = pd.to_numeric(df_temp["edad"], errors="coerce").fillna(0).clip(upper=80)
        if filtros["anio"] == "2015" and filtros["region"] == "Ñuble":
            st.warning("⚠️ No existen datos disponibles para la región de Ñuble en el año 2015.")
            return pd.DataFrame()
        if col not in df_temp.columns:
            df_temp = dia_tipo(df_temp, variables)
        df_plot = df_temp[["edad", col]].copy()
        df_plot.rename(columns={col: "Tiempo"}, inplace=True)
        df_plot["Tiempo"] = df_plot["Tiempo"].fillna(0)
        df_plot["Línea"] = etiqueta_linea
        df_plot["Detalle"] = f"{filtros['actividad']}"
        return df_plot
    df1 = preparar_df(df_15 if filtros_1["anio"] == "2015" else df_23, filtros_1, "Actividad N° 1")
    df2 = preparar_df(df_15 if filtros_2["anio"] == "2015" else df_23, filtros_2, "Actividad N° 2")
    df_final = pd.concat([df1, df2], ignore_index=True)
    df_final = df_final.groupby(["edad", "Línea", "Detalle"], as_index=False)["Tiempo"].mean()
    edades_completas = pd.DataFrame({'edad': list(range(0, 81))})
    lineas = df_final[["Línea", "Detalle"]].drop_duplicates()
    df_list = []
    for _, fila in lineas.iterrows():
        filtro = (df_final["Línea"] == fila["Línea"]) & (df_final["Detalle"] == fila["Detalle"])
        df_sub = df_final[filtro].merge(edades_completas, on="edad", how="right")
        df_sub["Línea"] = fila["Línea"]
        df_sub["Detalle"] = fila["Detalle"]
        df_sub["Tiempo"] = df_sub["Tiempo"].fillna(0)
        df_list.append(df_sub)
    df_final = pd.concat(df_list, ignore_index=True)
    df_final["Tiempohhmm"] = df_final["Tiempo"].apply(dec_a_hhmm)
    fig = px.line(
        df_final,
        x="edad",
        y="Tiempo",
        color="Línea",
        color_discrete_map={
            "Actividad N° 1": "#1f77b4",
            "Actividad N° 2": "#d62728",
        },
        labels={"edad": "Edad", "Tiempo": "Horas"},
        custom_data=["Tiempohhmm", "edad", "Detalle"],
        hover_data={"Tiempo": False, "Tiempohhmm": True, "edad": True, "Detalle": True},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Edad:</b> %{customdata[1]} años<br>"
            "<b>Actividad:</b> %{customdata[2]}<br>"
            "<b>Horas:</b> %{customdata[0]}<extra></extra>"
        )
    )
    fig.update_layout(
        xaxis=dict(range=[12, 80]),
        legend_title_text="Actividades"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        <div style='text-align: right; margin-top: -40px;'>
            <img src='https://i.imgur.com/wiPCMeE.png' width='50' style='border-radius: 5px;'>
        </div>
        """,
        unsafe_allow_html=True
    )
    
# ----------
# Función para graficar participación en actividades por sexo y región, con gráficos de barras y pie
# ----------
def graficar_participacion(df_15, df_23, variables, regiones, titulo1, titulo2):
    sexo_map = {1: "Hombres", 2: "Mujeres"}
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Besley&display=swap');
        .titulo-secundario {
            font-family: 'Besley', serif;
            font-size: 1.4em;
            color: black;
            font-weight: 600;
        }
        .subtitulo-secundario {
            font-family: Arial, sans-serif;
            font-size: 1.0em;
            color: black;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    cols1 = st.columns(2)
    with cols1[0]:
        anio = st.selectbox("Seleccione el año de interés:", ["2015", "2023"], index=1, key="version_participacion")
        df_base = df_15 if anio == "2015" else df_23
    with cols1[1]:
        region = st.selectbox("Seleccione región:", regiones, key="region_participacion")
    if anio == "2015" and region == "Ñuble":
        st.warning("⚠️ No existen datos disponibles para la región de Ñuble en el año 2015.")
        return
    if region != "Todas":
        df_base = df_base[df_base["id_reg"] == region]
    df_base = df_base[df_base["edad"] > 15].copy()
    df_base[variables] = df_base[variables].fillna(0)
    df_base["participa_actividad"] = df_base[variables].apply(lambda row: 1 if (row == 1).any() else 0, axis=1)
    resumen_sexo = df_base.groupby("sexo").agg(
        total_personas=("sexo", "count"),
        total_trabajan=("participa_actividad", "sum")
    ).reset_index()
    resumen_sexo["porcentaje_trabajan"] = 100 * resumen_sexo["total_trabajan"] / resumen_sexo["total_personas"]
    resumen_sexo["sexo_label"] = resumen_sexo["sexo"].map(sexo_map)
    fig_bar = px.bar(
        resumen_sexo,
        x="sexo_label",
        y="porcentaje_trabajan",
        labels={"sexo_label": "Sexo", "porcentaje_trabajan": "Porcentaje (%)"},
        color="sexo_label",
        color_discrete_map={"Hombres": "#5B9BD5", "Mujeres": "#F28AB2"},
        title=titulo1,
        text="porcentaje_trabajan"
    )
    fig_bar.update_layout(
        yaxis=dict(range=[0, 110])
    )
    fig_bar.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="outside",
        hovertemplate="<b>Sexo:</b> %{x}<br><b>Porcentaje:</b> %{y:.2f}%<extra></extra>"
    )
    df_trabajan = df_base[df_base["participa_actividad"] == 1]
    resumen_pie = df_trabajan["sexo"].value_counts(normalize=True).reset_index()
    resumen_pie.columns = ["sexo", "proporcion"]
    resumen_pie["sexo_label"] = resumen_pie["sexo"].map(sexo_map)
    fig_pie = px.pie(
        resumen_pie,
        names="sexo_label",
        values="proporcion",
        title=titulo2,
        color="sexo_label",
        category_orders={"sexo_label": ["Mujeres", "Hombres"]},
        color_discrete_map={"Hombres": "#5B9BD5", "Mujeres": "#F28AB2"},
    )
    fig_pie.update_traces(
        textinfo="percent",
        texttemplate="%{percent:.2%}",
        hovertemplate="<b>Porcentaje:</b> %{percent:.2%}<extra></extra>"
    )
    col2 = st.columns(2)
    with col2[0]:
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown(
            """
            <div style='text-align: right; margin-top: -40px;'>
                <img src='https://i.imgur.com/wiPCMeE.png' width='50' style='border-radius: 5px;'>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2[1]:
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown(
            """
            <div style='text-align: right; margin-top: -40px;'>
                <img src='https://i.imgur.com/wiPCMeE.png' width='50' style='border-radius: 5px;'>
            </div>
            """,
            unsafe_allow_html=True
        )
        
# ----------
# Gráfico piramidal de porcentaje de participación en actividades por edad y sexo
# ----------
def piramide_porcentaje(df_15, df_23, variables, regiones):
    cols1 = st.columns(2)
    with cols1[0]:
        anio = st.selectbox("Seleccione el año de interés:", ["2015", "2023"], index=1, key="version_piramide")
        df = df_15.copy() if anio == "2015" else df_23.copy()
    with cols1[1]:
        region = st.selectbox("Seleccione región:", regiones, key="region_piramide")
    if anio == "2015" and region == "Ñuble":
        st.warning("⚠️ No existen datos disponibles para la región de Ñuble en el año 2015.")
        return
    if region != "Todas":
        df = df[df["id_reg"] == region]
    df = df[df["edad"] > 15].copy()
    df["edad"] = df["edad"].apply(lambda x: 80 if x >= 80 else x)
    df["trabaja"] = 0
    sufijos = ["_ds", "_fds"]
    columnas_validas = [var + suf for var in variables for suf in sufijos if var + suf in df.columns]
    df[columnas_validas] = df[columnas_validas].fillna(0)
    for idx, row in df.iterrows():
        participa = False
        for var_base in variables:
            for suf in sufijos:
                var = var_base + suf
                if var in df.columns and row.get(var, 0) == 1:
                    participa = True
                    break
            if participa:
                break
        if participa:
            df.at[idx, "trabaja"] = 1
    df_grouped = df.groupby(["edad", "sexo"]).agg(
        total_personas=pd.NamedAgg(column="trabaja", aggfunc="count"),
        total_trabajan=pd.NamedAgg(column="trabaja", aggfunc="sum")
    ).reset_index()
    df_grouped["porcentaje"] = 100 * df_grouped["total_trabajan"] / df_grouped["total_personas"]
    edades = sorted(df_grouped["edad"].unique())
    porcentaje_hombres = []
    porcentaje_mujeres = []
    customdata_hombres = []
    customdata_mujeres = []
    for edad in edades:
        grupo_h = df_grouped[(df_grouped["edad"] == edad) & (df_grouped["sexo"] == 1)]
        grupo_m = df_grouped[(df_grouped["edad"] == edad) & (df_grouped["sexo"] == 2)]
        ph = grupo_h["porcentaje"].values[0] if not grupo_h.empty else 0
        pm = grupo_m["porcentaje"].values[0] if not grupo_m.empty else 0
        porcentaje_hombres.append(-ph)
        porcentaje_mujeres.append(pm)
        label_edad = f"{edad}" if edad == 80 else str(edad)
        customdata_hombres.append([label_edad, abs(ph)])
        customdata_mujeres.append([label_edad, pm])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=edades,
        x=porcentaje_hombres,
        name="Hombres",
        orientation='h',
        marker_color="#5B9BD5",
        customdata=customdata_hombres,
        hovertemplate="<b>Edad:</b> %{customdata[0]} años<br><b>Porcentaje:</b> %{customdata[1]:.2f}%<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        y=edades,
        x=porcentaje_mujeres,
        name="Mujeres",
        orientation='h',
        marker_color="#E26A6A",
        customdata=customdata_mujeres,
        hovertemplate="<b>Edad:</b> %{customdata[0]} años<br><b>Porcentaje:</b> %{customdata[1]:.2f}%<extra></extra>"
    ))
    fig.update_layout(
        barmode='overlay',
        xaxis=dict(
            title="Porcentaje",
            tickvals=[-100, -50, 0, 50, 100],
            ticktext=["100%", "50%", "0%", "50%", "100%"],
            range=[-100, 100],
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            showline=True,
            linecolor='black',
        ),
        yaxis=dict(
            title="Edad",
        ),
        bargap=0.1,
        legend=dict(x=0.7, y=1.1),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        """
        <div style='text-align: right; margin-top: -40px;'>
            <img src='https://i.imgur.com/wiPCMeE.png' width='50' style='border-radius: 5px;'>
        </div>
        """,
        unsafe_allow_html=True
    )
    
# ----------
# Diccionario de variables
# ----------

# ----------
# Diccionario de regiones para menú desplegable
# ----------
regiones = [
    "Todas",
    "Arica y Parinacota",
    "Tarapacá",
    "Antofagasta",
    "Atacama",
    "Coquimbo",
    "Valparaíso",
    "Región Metropolitana de Santiago",
    "Libertador General Bernardo O´Higgins",
    "Maule",
    "Ñuble",
    "Biobío",
    "La Araucanía",
    "Los Ríos",
    "Los Lagos",
    "Aysén del General Carlos Ibañez del Campo",
    "Magallanes y la Antártica Chilena",
]

# ----------
# Diccionario de variables: nombre legible por código de actividad
# ----------
nomb_act = {
    # Trabajo en la ocupación
    "to1_t": "Trabajo en la ocupación",
    "to2_t": "Traslado hacia el lugar de trabajo",
    "to3_t": "Traslado de regreso desde el lugar de trabajo",
    "to4_t": "Búsqueda de empleo o emprender",
    # Cuidados no remunerados
    "tc1_t": "Alimentar a otra persona",
    "tc2_t": "Acostar a otra persona",
    "tc3_t": "Asistir a otra persona en el baño",
    "tc4_t": "Bañar a otra persona",
    "tc5_t": "Vestir a otra persona",
    "tc6_t": "Administrar medicamentos a otra persona",
    "tc7_t": "Acompañar a otra persona a un centro de salud",
    "tc8_t": "Acompañar a otra persona a un establecimiento educacional",
    "tc9_t": "Ayudar con tareas escolares",
    "tc10_t": "Jugar o leer con otra persona",
    "tc11_t": "Acompañar a otra persona a su lugar de trabajo",
    # Trabajo doméstico no remunerado
    "td1_t": "Cocinar",
    "td2_t": "Poner y retirar la mesa",
    "td3_t": "Lavar la loza",
    "td4_t": "Limpiar la cocina",
    "td5_t": "Limpiar la vivienda",
    "td6_t": "Sacar la basura",
    "td7_t": "Lavar ropa",
    "td8_t": "Planchar y guardar ropa",
    "td9_t": "Reparar ropa",
    "td10_t": "Realizar reparaciones menores en la vivienda",
    "td11_t": "Reparar artículos del hogar o vehículos",
    "td12_t": "Pagar servicios",
    "td13_t": "Planificar cuentas o gastos del hogar",
    "td14_t": "Hacer compras para el hogar",
    "td15_t": "Cuidar mascotas",
    "td16_t": "Cuidar plantas",
    # Trabajo voluntario y ayuda a otros hogares
    "tv1_t": "Ayudar a otro hogar",
    "tv2_t": "Cuidar o asistir a personas de 0 a 14 años",
    "tv3_t": "Cuidar o asistir a personas de 15 a 65 años",
    "tv4_t": "Cuidar o asistir a personas de 66 años o más",
    "tv5_t": "Trabajar en una institución sin fines de lucro",
    "tv6_t": "Otras actividades de voluntariado",
    # Cuidados personales y actividades fisiológicas
    "cp1_t": "Dormir",
    "cp2_t": "Bañarse",
    "cp3_t": "Desayunar",
    "cp4_t": "Almorzar",
    "cp5_t": "Tomar once o cenar",
    "cp6_t": "Consulta con un profesional de salud",
    "cp7_t": "Traslado hacia una consulta médica",
    "cp8_t": "Traslado de regreso desde una consulta médica",
    # Educación
    "ed1_t": "Asistir a un establecimiento educacional",
    "ed2_t": "Traslado hacia un establecimiento educacional",
    "ed3_t": "Traslado de regreso desde un establecimiento educacional",
    "ed4_t": "Realizar tareas escolares",
    # Vida social, ocio y medios de comunicación
    "vs1_t": "Conversar con familiares o amistades",
    "vs2_t": "Asistir a eventos sociales",
    "vs3_t": "Participar en actividades religiosas",
    "vs4_t": "Realizar actividades artísticas",
    "vs5_t": "Jugar juegos de mesa o videojuegos",
    "vs6_t": "Practicar deportes",
    "vs7_t": "Leer",
    "vs8_t": "Ver televisión",
    "vs9_t": "Escuchar música, audios o podcasts",
    "vs10_t": "Utilizar medios de comunicación masiva",
    # Agregados principales
    "to": "Trabajo en la ocupación",
    "tcnr": "Cuidados no remunerados",
    "tdnr": "Trabajo doméstico no remunerado",
    "tvaoh": "Trabajo voluntario y ayuda a otros hogares",
    "cpaf": "Cuidados personales y actividades fisiológicas",
    "ed": "Educación",
    "vsyomcm": "Vida social, ocio y medios de comunicación",
    # Subagrupaciones por tipo
    "s_to": "Actividades del trabajo en la ocupación",
    "s_tto": "Traslados asociados al trabajo en la ocupación",
    "s_tcnr_ce": "Cuidados esenciales a personas del hogar",
    "s_tcnr_re": "Cuidados relacionados con la educación",
    "s_tcnr_oac": "Otras actividades de cuidado",
    "s_tdnr_psc": "Preparación y servicio de comidas",
    "s_tdnr_lv": "Limpieza del hogar",
    "s_tdnr_lrc": "Limpieza y reparación de ropa",
    "s_tdnr_mrm": "Mantención y reparaciones menores del hogar",
    "s_tdnr_admnhog": "Administración del hogar",
    "s_tdnr_comphog": "Compras para el hogar",
    "s_tdnr_cmp": "Cuidado de mascotas y plantas",
    "s_tvaoh_tv": "Voluntariado en ISFL o comunidad",
    "s_tvaoh_oh": "Ayuda a otros hogares",
    "s_cpaf_cp": "Cuidados personales",
    "s_cpaf_af": "Actividades fisiológicas",
    "s_ed": "Actividades educativas",
    "s_ted": "Traslados asociados a la educación",
    "s_vsyo": "Actividades de vida social y ocio",
    "s_mcm": "Medios de comunicación masiva",
    # Variables internas
    "tnr": "Trabajo no remunerado",
    "cpaf_sin_cp1": "Cuidados personales",
    "tt": "Traslados"
}

# ----------
# Diccionario de páginas: nombre visible en la plataforma
# ----------
pag = {
    "PP"     : "Página principal",
    "UTD"    : "Uso del tiempo diario",
    "TO"     : "Trabajo en la ocupación",
    "TCNR"   : "Cuidados de terceros",
    "TDNR"   : "Trabajo doméstico",
    "TVAOH"  : "Trabajo voluntario",
    "CPAF"   : "Cuidados personales",
    "ED"     : "Educación",
    "VSYOMCM": "Vida social y ocio"
}

# ----------
# Carga de bases de datos 2015 y 2023
# ----------
df_15, df_23 = Cargando_Datos()

# ----------
# Controlador de páginas
# ----------
if "pagina" not in st.session_state:
    st.session_state.pagina = "PP"

st.sidebar.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700;800&display=swap');
        .stSidebar h2 {
            font-family: 'Playfair Display', serif;
            font-size: 20px;
            font-weight: 600;
            color: #4b2995;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## Menú de navegación:")

for nombre, etiqueta in pag.items():
    if st.sidebar.button(etiqueta, key=nombre, use_container_width=True):
        st.session_state.pagina = nombre

pagina = st.session_state.pagina

# ----------
# Página principal
# ----------
if pagina == "PP":
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Playfair+Display:wght@700&display=swap');
        .encabezado-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 25px;
            background-color: #F9F7FC;
            border-left: 6px solid #B57EDC;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            max-width: 100%;
            margin-bottom: 15px;
        }        
        .titulo-principal {
            font-family: 'Playfair Display', serif;
            color: #4B2995;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .subtitulo-principal {
            font-family: 'Montserrat', sans-serif;
            color: #7F56D9;
            font-size: 1.0em;
            font-weight: 500;
        }
        </style>

        <div class="encabezado-container">
            <div class="titulo-principal">Encuesta Nacional sobre Uso del Tiempo</div>
            <div class="subtitulo-principal">Descubre cómo las personas en Chile organizan su tiempo día a día</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Besley&display=swap');
        .titulo-secundario {
            font-family: 'Besley', serif;
            font-size: 1.4em;
            color: black;
            font-weight: 600;
        }
        .subtitulo-secundario {
            font-family: Arial, sans-serif;
            font-size: 1.0em;
            color: gray;
            font-weight: 500;
        }
        </style>

    <div class="titulo-secundario">Distribución del tiempo durante el día</div>
    <div class="subtitulo-secundario">Promedio diario del tiempo dedicado a cada actividad según la edad</div>
        """,
        unsafe_allow_html=True
    )

    df = df_23.copy()

    variables_gdtld = ['s_to', 's_tto', 'tdnr', 'tcnr', 'tvaoh', 'cp1_t', 'cpaf', 's_ed', 's_ted', 'vsyomcm']
    df = dia_tipo(df, variables_gdtld)

    df["tt_dt"] = df["s_tto_dt"] + df["s_ted_dt"]
    df["tnr_dt"] = df["tdnr_dt"] + df["tcnr_dt"] + df["tvaoh_dt"]
    df["cpaf_sin_cp1_dt"] = df["cpaf_dt"] - df["cp1_t_dt"]

    col_utdd = [f"{v}_dt" for v in variables_gdtld if v not in ['s_tto', 'tdnr', 'tcnr', 'tvaoh', 'cpaf', 's_ted']]
    col_utdd.extend(["tt_dt", "tnr_dt", "cpaf_sin_cp1_dt"])

    edad_col = "edad"
    df[edad_col] = pd.to_numeric(df[edad_col], errors='coerce')
    df[edad_col] = df[edad_col].apply(lambda x: 80 if x >= 80 else x)
    df_prom_act_edad = df.groupby(edad_col)[col_utdd].mean().reset_index()

    df_long_act_edad = pd.melt(
        df_prom_act_edad,
        id_vars=edad_col,
        value_vars=col_utdd,
        var_name='Actividad',
        value_name='Tiempo'
    )

    df_long_act_edad["Actividad"] = df_long_act_edad["Actividad"].str.replace("_dt", "")

    nombres_grafico = {
        "s_to": "Trabajo en la ocupación",
        "tt": "Traslados",
        "cpaf_sin_cp1": "Cuidados personales",
        "s_ed": "Educación",
        "cp1_t": "Horas de sueño",
        "vsyomcm": "Vida social y ocio",
        "tnr": "Trabajo no remunerado"
    }

    df_long_act_edad["Actividad"] = df_long_act_edad["Actividad"].map(nombres_grafico).fillna(df_long_act_edad["Actividad"])

    df_long_act_edad = df_long_act_edad.sort_values(by=edad_col)
    df_long_act_edad["Tiempo"] = pd.to_numeric(df_long_act_edad["Tiempo"], errors='coerce').fillna(0)
    df_long_act_edad["Tiempohhmm"] = df_long_act_edad["Tiempo"].apply(dec_a_hhmm)

    orden_act = [
        "Trabajo en la ocupación",
        "Trabajo no remunerado",
        "Horas de sueño",
        "Cuidados personales",
        "Educación",
        "Traslados",
        "Vida social y ocio"
    ]
    df_long_act_edad["Actividad"] = pd.Categorical(df_long_act_edad["Actividad"], categories=orden_act, ordered=True)
    df_long_act_edad = df_long_act_edad.sort_values([edad_col, "Actividad"])

    fig_act_dia = px.area(
        df_long_act_edad,
        x=edad_col,
        y="Tiempo",
        color="Actividad",
        labels={edad_col: 'Edad', 'Tiempo': 'Horas'},
        custom_data=["Tiempohhmm", edad_col, "Actividad"],
        hover_data={"Tiempo": False, "Tiempohhmm": True, edad_col: True, "Actividad": True}
    )

    fig_act_dia.update_traces(
        hovertemplate="<b>Edad:</b> %{customdata[1]}<br>" +
                      "<b>Actividad:</b> %{customdata[2]}<br>" +
                      "<b>Horas:</b> %{customdata[0]}<extra></extra>"
    )

    fig_act_dia.update_layout(
        xaxis=dict(range=[12, df_long_act_edad[edad_col].max()]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=700,
        height=500,
        yaxis=dict(range=[0, 24]),
        legend=dict(
            orientation="h",
            y=-0.3,
            x=0.5,
            xanchor='center'
        )
    )

    st.plotly_chart(fig_act_dia, use_container_width=True)

    st.markdown(
        """
        <div style='text-align: right; margin-top: -15px;'>
            <img src='https://i.imgur.com/wiPCMeE.png' width='50' style='border-radius: 5px;'>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    # ----------
    # Gráfica porcentaje por género (2023)
    # ----------
    st.markdown(
        """
        <div class="titulo-secundario">Participación en trabajo remunerado y no remunerado según sexo</div>
        <div class="subtitulo-secundario">Porcentaje de personas de 15 años o más que realizan actividades de trabajo remunerado o no remunerado, desagregado por sexo</div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        var_pto = ["to1_p_ds", "to1_p_fds", "to4_p_ds", "to4_p_fds"]
        titulo = "Personas que trabajan o buscan activamente empleo"
        grafico_barras(df, var_pto, titulo)
    with col2:
        var_ptcnr = [
            "tc1_p_ds", "tc1_p_fds", "tc2_p_ds", "tc2_p_fds", "tc3_p_ds", "tc3_p_fds",
            "tc4_p_ds", "tc4_p_fds", "tc5_p_ds", "tc5_p_fds", "tc6_p_ds", "tc6_p_fds",
            "tc7_p_ds", "tc7_p_fds", "tc8_p_ds", "tc8_p_fds", "tc9_p_ds", "tc9_p_fds",
            "tc10_p_ds", "tc10_p_fds", "tc11_p_ds", "tc11_p_fds"
        ]
        titulo = "Personas que realizan labores de cuidado de terceros"
        grafico_barras(df, var_ptcnr, titulo)
    
    col3, col4 = st.columns(2)
    with col3:
        var_ptdnr = [
            "td1_p_ds", "td1_p_fds", "td2_p_ds", "td2_p_fds", "td3_p_ds", "td3_p_fds",
            "td4_p_ds", "td4_p_fds", "td5_p_ds", "td5_p_fds", "td6_p_ds", "td6_p_fds",
            "td7_p_ds", "td7_p_fds", "td8_p_ds", "td8_p_fds", "td9_p_ds", "td9_p_fds",
            "td10_p_ds", "td10_p_fds", "td11_p_ds", "td11_p_fds", "td12_p_ds", "td12_p_fds",
            "td13_p_ds", "td13_p_fds", "td14_p_ds", "td14_p_fds", "td15_p_ds", "td15_p_fds",
            "td16_p_ds", "td16_p_fds"
        ]
        titulo = "Personas que realizan trabajo doméstico en su propio hogar"
        grafico_barras(df, var_ptdnr, titulo)
    with col4:
        var_ptvaoh = [
            "tv1_p_ds", "tv1_p_fds", "tv2_p_ds", "tv2_p_fds", "tv3_p_ds", "tv3_p_fds",
            "tv4_p_ds", "tv4_p_fds", "tv5_p_ds", "tv5_p_fds", "tv6_p_ds", "tv6_p_fds"
        ]
        titulo = "Personas que realizan trabajo voluntario"
        grafico_barras(df, var_ptvaoh, titulo)
        
    # ----------
    # Gráfica comparativa de uso del tiempo para todas actividades
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Compara el tiempo dedicado a actividades claves</div>
        <div class="subtitulo-secundario">Explora y compara actividades de tu interés según año, sexo, día y región</div>
        """,
        unsafe_allow_html=True
    )

    variables_PP = [
        "to1_t",  "to2_t",  "to3_t",  "to4_t",
        "tc1_t",  "tc2_t",  "tc3_t",  "tc4_t",
        "tc5_t",  "tc6_t",  "tc7_t",  "tc8_t",
        "tc9_t", "tc10_t", "tc11_t",
        "td1_t",  "td2_t",  "td3_t",  "td4_t",
        "td5_t",  "td6_t",  "td7_t",  "td8_t",
        "td9_t", "td10_t", "td11_t", "td12_t",
        "td13_t", "td14_t", "td15_t", "td16_t",
        "tv1_t",  "tv2_t",  "tv3_t",  "tv4_t",
        "tv5_t",  "tv6_t",
        "cp1_t",  "cp2_t",  "cp3_t",  "cp4_t",
        "cp5_t",  "cp6_t",  "cp7_t",  "cp8_t",
        "ed1_t",  "ed2_t",  "ed3_t",  "ed4_t",
        "vs1_t",  "vs2_t",  "vs3_t",  "vs4_t",
        "vs5_t",  "vs6_t",  "vs7_t",  "vs8_t",
        "vs9_t", "vs10_t"
    ]

    comparar_actividades(df_15, df_23, variables_PP, nomb_act, regiones)
    
# ----------
# Página Uso del Tiempo Diario
# ----------
elif pagina == "UTD":
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Playfair+Display:wght@700&display=swap');
        .encabezado-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 25px;
            background-color: #F9F7FC;
            border-left: 6px solid #B57EDC;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            max-width: 100%;
            margin-bottom: 15px;
        }        
        .titulo-principal {
            font-family: 'Playfair Display', serif;
            color: #4B2995;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .subtitulo-principal {
            font-family: 'Montserrat', sans-serif;
            color: #7F56D9;
            font-size: 1.0em;
            font-weight: 500;
        }
        </style>

        <div class="encabezado-container">
            <div class="titulo-principal">¿Cómo organizamos nuestro día?</div>
            <div class="subtitulo-principal">Principales resultados sobre el uso del tiempo en Chile, agrupados en grandes categorías de actividades</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    variables_UTD = ["to", "tcnr", "tdnr", "tvaoh", "cpaf", "ed", "vsyomcm"]
    
    # ----------
    # Gráfica de área apilada por edad (ENUT 2015 y 2023)
    # ----------
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Besley&display=swap');
        .titulo-secundario {
            font-family: 'Besley', serif;
            font-size: 1.4em;
            color: black;
            font-weight: 600;
        }
        .subtitulo-secundario {
            font-family: Arial, sans-serif;
            font-size: 1.0em;
            color: gray;
            font-weight: 500;
        }
        </style>

    <div class="titulo-secundario">Distribución del tiempo durante el día</div>
    <div class="subtitulo-secundario">Promedio diario del tiempo dedicado a cada actividad según la edad</div>
        """,
        unsafe_allow_html=True
    )

    graficar_area_apilada(df_15, df_23, variables_UTD, nomb_act, regiones)

    # ----------
    # Gráfica comparativa de actividades específicas
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Compara el tiempo dedicado a grandes grupos de actividades</div>
        <div class="subtitulo-secundario">Explora y compara actividades de tu interés según año, sexo, día y región</div>
        """,
        unsafe_allow_html=True
    )

    comparar_actividades(df_15, df_23, variables_UTD, nomb_act, regiones)
    
# ----------
# Página Trabajo en la Ocupación
# ----------
elif pagina == "TO":
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Playfair+Display:wght@700&display=swap');
        .encabezado-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 25px;
            background-color: #F9F7FC;
            border-left: 6px solid #B57EDC;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            max-width: 100%;
            margin-bottom: 15px;
        }        
        .titulo-principal {
            font-family: 'Playfair Display', serif;
            color: #4B2995;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .subtitulo-principal {
            font-family: 'Montserrat', sans-serif;
            color: #7F56D9;
            font-size: 1.0em;
            font-weight: 500;
        }
        </style>

        <div class="encabezado-container">
            <div class="titulo-principal">Trabajo en la ocupación</div>
            <div class="subtitulo-principal">Tiempo que dedican las personas al trabajo remunerado</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    variables_TO1 = ["s_to", "s_tto"]
    variables_TO2 = ["to1_t", "to2_t", "to3_t", "to4_t"]
    variables_TO3 = ["to1_p_ds", "to1_p_fds", "to4_p_ds", "to4_p_fds"]
    variables_TO4 = ["to1_p", "to4_p"]

    # ----------
    # Gráfico de área apilada
    # ----------
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Besley&display=swap');
        .titulo-secundario {
            font-family: 'Besley', serif;
            font-size: 1.4em;
            color: black;
            font-weight: 600;
        }
        .subtitulo-secundario {
            font-family: Arial, sans-serif;
            font-size: 1.0em;
            color: gray;
            font-weight: 500;
        }
        </style>
        <div class="titulo-secundario">Distribución del tiempo durante el día</div>
        <div class="subtitulo-secundario">Promedio diario del tiempo dedicado a cada actividad según la edad</div>
        """,
        unsafe_allow_html=True
    )
    
    graficar_area_apilada(df_15, df_23, variables_TO1, nomb_act, regiones)

    # ----------
    # Gráficos de participación
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Participación en actividades laborales remuneradas según sexo</div>
        <div class="subtitulo-secundario">Porcentaje de la población de 15 años y más que participa en el mercado laboral, ya sea trabajando o buscando activamente empleo</div>
        """,
        unsafe_allow_html=True
    )

    graficar_participacion(df_15, df_23, variables_TO3, regiones, "Participación laboral", "Distribución de personas ocupadas")

    # ----------
    # Gráfico de pirámide por edad y sexo
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Participación en actividades laborales remuneradas según sexo y edad</div>
        <div class="subtitulo-secundario">Porcentaje de la población de 15 años y más que participa en el mercado laboral, ya sea trabajando o buscando activamente empleo</div>
        """,
        unsafe_allow_html=True
    )

    piramide_porcentaje(df_15, df_23, variables_TO4, regiones)

    # ----------
    # Gráfico comparativo de dos actividades
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Compara el tiempo dedicado al trabajo en la ocupación</div>
        <div class="subtitulo-secundario">Explora y compara actividades de tu interés según año, sexo, día y región</div>
        """,
        unsafe_allow_html=True
    )

    comparar_actividades(df_15, df_23, variables_TO2, nomb_act, regiones)
    
# ----------
# Página Trabajo de Cuidados No Remunerado de Cuidados
# ----------
elif pagina == "TCNR":
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Playfair+Display:wght@700&display=swap');
        .encabezado-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 25px;
            background-color: #F9F7FC;
            border-left: 6px solid #B57EDC;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            max-width: 100%;
            margin-bottom: 15px;
        }        
        .titulo-principal {
            font-family: 'Playfair Display', serif;
            color: #4B2995;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .subtitulo-principal {
            font-family: 'Montserrat', sans-serif;
            color: #7F56D9;
            font-size: 1.0em;
            font-weight: 500;
        }
        </style>

        <div class="encabezado-container">
            <div class="titulo-principal">Cuidados de terceros</div>
            <div class="subtitulo-principal">Tiempo que dedican las personas al cuidado no remunerado de niños, adultos mayores u otras personas del hogar</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    variables_TCNR1 = ["s_tcnr_ce", "s_tcnr_re", "s_tcnr_oac"]
    variables_TCNR2 = [
        "tc1_t", "tc2_t", "tc3_t", "tc4_t", "tc5_t", "tc6_t", "tc7_t", 
        "tc8_t", "tc9_t", "tc10_t", "tc11_t"
    ]
    variables_TCNR3 = [
        "tc1_p_ds", "tc2_p_ds", "tc3_p_ds", "tc4_p_ds", "tc5_p_ds", "tc6_p_ds", "tc7_p_ds", 
        "tc8_p_ds", "tc9_p_ds", "tc10_p_ds", "tc11_p_ds",
        "tc1_p_fds", "tc2_p_fds", "tc3_p_fds", "tc4_p_fds", "tc5_p_fds", "tc6_p_fds", "tc7_p_fds", 
        "tc8_p_fds", "tc9_p_fds", "tc10_p_fds", "tc11_p_fds"
    ]
    variables_TCNR4 = [
        "tc1_p", "tc2_p", "tc3_p", "tc4_p", "tc5_p", "tc6_p", "tc7_p", 
        "tc8_p", "tc9_p", "tc10_p", "tc11_p"
    ]

    # ----------
    # Gráfico de área apilada
    # ----------
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Besley&display=swap');
        .titulo-secundario {
            font-family: 'Besley', serif;
            font-size: 1.4em;
            color: black;
            font-weight: 600;
        }
        .subtitulo-secundario {
            font-family: Arial, sans-serif;
            font-size: 1.0em;
            color: gray;
            font-weight: 500;
        }
        </style>
        <div class="titulo-secundario">Distribución del tiempo durante el día</div>
        <div class="subtitulo-secundario">Promedio diario del tiempo dedicado a cada actividad según la edad</div>
        """,
        unsafe_allow_html=True
    )
    
    graficar_area_apilada(df_15, df_23, variables_TCNR1, nomb_act, regiones)

    # ----------
    # Gráfico de torta
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Participación en actividades de cuidado no remunerado según sexo</div>
        <div class="subtitulo-secundario">Porcentaje de la población de 15 años y más que realiza cuidados no remunerados a niños, adultos mayores u otros miembros del hogar</div>
        """,
        unsafe_allow_html=True
    )

    graficar_participacion(df_15, df_23, variables_TCNR3, regiones, "Participación en cuidados a terceros", "Distribución de personas dedicadas a cuidados a terceros")

    # ----------
    # Gráfico pirámide
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Participación en actividades de cuidado no remunerado según sexo y edad</div>
        <div class="subtitulo-secundario">Porcentaje de la población de 15 años y más que realiza cuidados no remunerados a niños, adultos mayores u otros miembros del hogar</div>
        """,
        unsafe_allow_html=True
    )

    piramide_porcentaje(df_15, df_23, variables_TCNR4, regiones)

    # ----------
    # Gráfico comparativo de dos actividades clave
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Compara el tiempo dedicado a los cuidados de terceros</div>
        <div class="subtitulo-secundario">Explora y compara actividades de tu interés según año, sexo, día y región</div>
        """,
        unsafe_allow_html=True
    )

    comparar_actividades(df_15, df_23, variables_TCNR2, nomb_act, regiones)
    
# ----------
# Página Tiempo en Trabajos Domésticos No Remunerados
# ----------
elif pagina == "TDNR":
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Playfair+Display:wght@700&display=swap');
        .encabezado-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 25px;
            background-color: #F9F7FC;
            border-left: 6px solid #B57EDC;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            max-width: 100%;
            margin-bottom: 15px;
        }        
        .titulo-principal {
            font-family: 'Playfair Display', serif;
            color: #4B2995;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .subtitulo-principal {
            font-family: 'Montserrat', sans-serif;
            color: #7F56D9;
            font-size: 1.0em;
            font-weight: 500;
        }
        </style>

        <div class="encabezado-container">
            <div class="titulo-principal">Trabajo doméstico</div>
            <div class="subtitulo-principal">Tiempo que dedican las personas a realizar tareas del hogar</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    variables_TDNR1 = [
        "s_tdnr_psc", "s_tdnr_lv", "s_tdnr_lrc", "s_tdnr_mrm", 
        "s_tdnr_admnhog", "s_tdnr_comphog", "s_tdnr_cmp"
    ]
    variables_TDNR2 = [
        "td1_t", "td2_t", "td3_t", "td4_t", "td5_t", "td6_t", "td7_t", "td8_t", 
        "td9_t", "td10_t", "td11_t", "td12_t", "td13_t", "td14_t", "td15_t", "td16_t"
    ]
    variables_TDNR3 = [
        "td1_p_ds", "td2_p_ds", "td3_p_ds", "td4_p_ds", "td5_p_ds", "td6_p_ds", "td7_p_ds", "td8_p_ds", 
        "td9_p_ds", "td10_p_ds", "td11_p_ds", "td12_p_ds", "td13_p_ds", "td14_p_ds", "td15_p_ds", "td16_p_ds",
        "td1_p_fds", "td2_p_fds", "td3_p_fds", "td4_p_fds", "td5_p_fds", "td6_p_fds", "td7_p_fds", "td8_p_fds", 
        "td9_p_fds", "td10_p_fds", "td11_p_fds", "td12_p_fds", "td13_p_fds", "td14_p_fds", "td15_p_fds", "td16_p_fds"
    ]
    variables_TDNR4 = [
        "td1_p", "td2_p", "td3_p", "td4_p", "td5_p", "td6_p", "td7_p", "td8_p", 
        "td9_p", "td10_p", "td11_p", "td12_p", "td13_p", "td14_p", "td15_p", "td16_p"
    ]

    # ----------
    # Gráfico de área apilada
    # ----------
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Besley&display=swap');
        .titulo-secundario {
            font-family: 'Besley', serif;
            font-size: 1.4em;
            color: black;
            font-weight: 600;
        }
        .subtitulo-secundario {
            font-family: Arial, sans-serif;
            font-size: 1.0em;
            color: gray;
            font-weight: 500;
        }
        </style>
        <div class="titulo-secundario">Distribución del tiempo durante el día</div>
        <div class="subtitulo-secundario">Promedio diario del tiempo dedicado a cada actividad según la edad</div>
        """,
        unsafe_allow_html=True
    )

    graficar_area_apilada(df_15, df_23, variables_TDNR1, nomb_act, regiones)

    # ----------
    # Gráfico de torta
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Participación en actividades domésticas no remuneradas según sexo</div>
        <div class="subtitulo-secundario">Porcentaje de la población de 15 años y más que realiza tareas domésticas</div>
        """,
        unsafe_allow_html=True
    )
    
    graficar_participacion(df_15, df_23, variables_TDNR3, regiones, "Participación en trabajo doméstico", "Distribución del trabajo doméstico")
    
    # ----------
    # Gráfico pirámide
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Participación en actividades domésticas no remuneradas según sexo y edad</div>
        <div class="subtitulo-secundario">Porcentaje de la población de 15 años y más que realiza tareas domésticas</div>
        """,
        unsafe_allow_html=True
    )
    piramide_porcentaje(df_15, df_23, variables_TDNR4, regiones)
    
    # ----------
    # Gráfico comparativo de 2 actividades
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Compara el tiempo dedicado al trabajo doméstico</div>
        <div class="subtitulo-secundario">Explora y compara actividades de tu interés según año, sexo, día y región</div>
        """,
        unsafe_allow_html=True
    )
    comparar_actividades(df_15, df_23, variables_TDNR2, nomb_act, regiones)
    
# ----------
# Página Trabajo Voluntario y Ayuda a Otros Hogares
# ----------
elif pagina == "TVAOH":
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Playfair+Display:wght@700&display=swap');
        .encabezado-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 25px;
            background-color: #F9F7FC;
            border-left: 6px solid #B57EDC;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            max-width: 100%;
            margin-bottom: 15px;
        }        
        .titulo-principal {
            font-family: 'Playfair Display', serif;
            color: #4B2995;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .subtitulo-principal {
            font-family: 'Montserrat', sans-serif;
            color: #7F56D9;
            font-size: 1.0em;
            font-weight: 500;
        }
        </style>

        <div class="encabezado-container">
            <div class="titulo-principal">Trabajo voluntario</div>
            <div class="subtitulo-principal">Tiempo que dedican las personas a actividades voluntarias o de ayuda a otros hogares</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")

    variables_TVAOH1 = ["s_tvaoh_tv", "s_tvaoh_oh"]
    variables_TVAOH2 = ["tv1_t", "tv2_t", "tv3_t", "tv4_t", "tv5_t", "tv6_t"]
    variables_TVAOH3 = [
        "tv1_p_ds", "tv2_p_ds", "tv3_p_ds", "tv4_p_ds", "tv5_p_ds", "tv6_p_ds",
        "tv1_p_fds", "tv2_p_fds", "tv3_p_fds", "tv4_p_fds", "tv5_p_fds", "tv6_p_fds"
    ]
    variables_TVAOH4 = ["tv1_p", "tv2_p", "tv3_p", "tv4_p", "tv5_p", "tv6_p"]
    
    # ----------
    # Gráfico de área apilada
    # ----------
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Besley&display=swap');
        .titulo-secundario {
            font-family: 'Besley', serif;
            font-size: 1.4em;
            color: black;
            font-weight: 600;
        }
        .subtitulo-secundario {
            font-family: Arial, sans-serif;
            font-size: 1.0em;
            color: gray;
            font-weight: 500;
        }
        </style>
        <div class="titulo-secundario">Distribución del tiempo durante el día</div>
        <div class="subtitulo-secundario">Promedio diario del tiempo dedicado a cada actividad según la edad</div>
        """,
        unsafe_allow_html=True
    )

    graficar_area_apilada(df_15, df_23, variables_TVAOH1, nomb_act, regiones)

    # ----------
    # Gráfico de torta
    # ----------
    st.markdown(
        """
        <div class="titulo-secundario">Participación en actividades de trabajo voluntario según sexo</div>
        <div class="subtitulo-secundario">Porcentaje de la población de 15 años y más que participa en actividades voluntarias o de apoyo comunitario</div>
        """,
        unsafe_allow_html=True
    )
    
    graficar_participacion(df_15, df_23, variables_TVAOH3, regiones, "Participación en actividades voluntarias", "Distribución de personas que realizan trabajo voluntario")

    # ----------
    # Gráfico pirámide
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Participación en actividades de trabajo voluntario según sexo y edad</div>
        <div class="subtitulo-secundario">Porcentaje de la población de 15 años y más que participa en actividades voluntarias o de apoyo comunitario</div>
        """,
        unsafe_allow_html=True
    )
    
    piramide_porcentaje(df_15, df_23, variables_TVAOH4, regiones)

    # ----------
    # Gráfico comparativo de 2 actividades
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Compara el tiempo dedicado al trabajo voluntario</div>
        <div class="subtitulo-secundario">Explora y compara actividades de tu interés según año, sexo, día y región</div>
        """,
        unsafe_allow_html=True
    )
    
    comparar_actividades(df_15, df_23, variables_TVAOH2, nomb_act, regiones)
    
# ----------
# Página Tiempo dedicado a cuidados personales
# ----------
elif pagina == "CPAF":
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Playfair+Display:wght@700&display=swap');
        .encabezado-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 25px;
            background-color: #F9F7FC;
            border-left: 6px solid #B57EDC;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            max-width: 100%;
            margin-bottom: 15px;
        }        
        .titulo-principal {
            font-family: 'Playfair Display', serif;
            color: #4B2995;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .subtitulo-principal {
            font-family: 'Montserrat', sans-serif;
            color: #7F56D9;
            font-size: 1.0em;
            font-weight: 500;
        }
        </style>

        <div class="encabezado-container">
            <div class="titulo-principal">Cuidados personales</div>
            <div class="subtitulo-principal">Tiempo que dedican las personas a su bienestar personal</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")

    variables_CPAF1 = ["s_cpaf_cp", "s_cpaf_af"]
    variables_CPAF2 = ["cp1_t", "cp2_t", "cp3_t", "cp4_t", "cp5_t", "cp6_t", "cp7_t", "cp8_t"]

    # ----------
    # Gráfico de área apilada
    # ----------
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Besley&display=swap');
        .titulo-secundario {
            font-family: 'Besley', serif;
            font-size: 1.4em;
            color: black;
            font-weight: 600;
        }
        .subtitulo-secundario {
            font-family: Arial, sans-serif;
            font-size: 1.0em;
            color: gray;
            font-weight: 500;
        }
        </style>
        <div class="titulo-secundario">Distribución del tiempo durante el día</div>
        <div class="subtitulo-secundario">Promedio diario del tiempo dedicado a cada actividad según la edad</div>
        """,
        unsafe_allow_html=True
    )

    graficar_area_apilada(df_15, df_23, variables_CPAF1, nomb_act, regiones)

    # ----------
    # Gráfico comparativo de 2 actividades
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Compara el tiempo dedicado a los cuidados personales</div>
        <div class="subtitulo-secundario">Explora y compara actividades de tu interés según año, sexo, día y región</div>
        """,
        unsafe_allow_html=True
    )
    
    comparar_actividades(df_15, df_23, variables_CPAF2, nomb_act, regiones)
    
# ----------
# Página Tiempo dedicado a la educación
# ----------
elif pagina == "ED":
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Playfair+Display:wght@700&display=swap');
        .encabezado-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 25px;
            background-color: #F9F7FC;
            border-left: 6px solid #B57EDC;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            max-width: 100%;
            margin-bottom: 15px;
        }        
        .titulo-principal {
            font-family: 'Playfair Display', serif;
            color: #4B2995;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .subtitulo-principal {
            font-family: 'Montserrat', sans-serif;
            color: #7F56D9;
            font-size: 1.0em;
            font-weight: 500;
        }
        </style>

        <div class="encabezado-container">
            <div class="titulo-principal">Educación</div>
            <div class="subtitulo-principal">Tiempo que dedican las personas a actividades educativas</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    variables_ED1 = ["s_ed", "s_ted"]
    variables_ED2 = ["ed1_t", "ed2_t", "ed3_t", "ed4_t"]

    # ----------
    # Gráfico de área apilada
    # ----------
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Besley&display=swap');
        .titulo-secundario {
            font-family: 'Besley', serif;
            font-size: 1.4em;
            color: black;
            font-weight: 600;
        }
        .subtitulo-secundario {
            font-family: Arial, sans-serif;
            font-size: 1.0em;
            color: gray;
            font-weight: 500;
        }
        </style>
        <div class="titulo-secundario">Distribución del tiempo durante el día</div>
        <div class="subtitulo-secundario">Promedio diario del tiempo dedicado a cada actividad según la edad</div>
        """,
        unsafe_allow_html=True
    )

    graficar_area_apilada(df_15, df_23, variables_ED1, nomb_act, regiones)

    # ----------
    # Gráfico comparativo de 2 actividades
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Compara el tiempo dedicado a la educación</div>
        <div class="subtitulo-secundario">Explora y compara actividades de tu interés según año, sexo, día y región</div>
        """,
        unsafe_allow_html=True
    )
    
    comparar_actividades(df_15, df_23, variables_ED2, nomb_act, regiones)
    
# ----------
# Página Tiempo dedicado a la vida social y ocio
# ----------
elif pagina == "VSYOMCM":
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Playfair+Display:wght@700&display=swap');
        .encabezado-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            padding: 15px 25px;
            background-color: #F9F7FC;
            border-left: 6px solid #B57EDC;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            max-width: 100%;
            margin-bottom: 15px;
        }        
        .titulo-principal {
            font-family: 'Playfair Display', serif;
            color: #4B2995;
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .subtitulo-principal {
            font-family: 'Montserrat', sans-serif;
            color: #7F56D9;
            font-size: 1.0em;
            font-weight: 500;
        }
        </style>

        <div class="encabezado-container">
            <div class="titulo-principal">Vida social y ocio</div>
            <div class="subtitulo-principal">Tiempo que dedican las personas a actividades recreativas, sociales y de uso de medios de comunicación</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    variables_VSYOMCM1 = ["s_vsyo", "s_mcm"]
    variables_VSYOMCM2 = [
        "vs1_t", "vs2_t", "vs3_t", "vs4_t", "vs5_t", 
        "vs6_t", "vs7_t", "vs8_t", "vs9_t", "vs10_t"
    ]

    # ----------
    # Gráfico de área apilada
    # ----------
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Besley&display=swap');
        .titulo-secundario {
            font-family: 'Besley', serif;
            font-size: 1.4em;
            color: black;
            font-weight: 600;
        }
        .subtitulo-secundario {
            font-family: Arial, sans-serif;
            font-size: 1.0em;
            color: gray;
            font-weight: 500;
        }
        </style>
        <div class="titulo-secundario">Distribución del tiempo durante el día</div>
        <div class="subtitulo-secundario">Promedio diario del tiempo dedicado a cada actividad según la edad</div>
        """,
        unsafe_allow_html=True
    )
    graficar_area_apilada(df_15, df_23, variables_VSYOMCM1, nomb_act, regiones)

    # ----------
    # Gráfico comparativo de 2 actividades
    # ----------
    st.markdown("---")
    st.markdown(
        """
        <div class="titulo-secundario">Compara el tiempo dedicado a la vida social y ocio</div>
        <div class="subtitulo-secundario">Explora y compara actividades de tu interés según año, sexo, día y región</div>
        """,
        unsafe_allow_html=True
    )
    
    comparar_actividades(df_15, df_23, variables_VSYOMCM2, nomb_act, regiones)
