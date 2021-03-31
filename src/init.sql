create table jobs
(
	uid text
		constraint jobs_pk
			primary key,
	name text not null,
	status text not null,
	file_path text not null
);

create unique index jobs_uid_uindex
	on jobs (uid);
